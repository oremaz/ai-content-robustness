import hashlib
import json

import pytest
import torch

from ai_content_detector.rl_evasion.checkpointing import atomic_save, load_checkpoint
from scripts import architecture_races as races


def completed_pilot(tmp_path, monkeypatch):
    monkeypatch.setattr(races, "OUTPUT", tmp_path)
    monkeypatch.setenv("USER", "fixture-user")
    training = {"learning_rate": 5e-5, "lr_scheduler_type": "constant_with_warmup", "warmup_steps": 50}
    protocol = {"attacker_steps_per_round": 300, "defenders": races.DEFENDERS,
                "execution": {"attacker_training": training}}
    manifest = {"protocol": protocol, "pause_after_attacker_round": 1, "inputs": {},
                "baseline_plan": {}, "jobs": {}}
    for name in races.DEFENDERS:
        directory = tmp_path / "source" / name
        attacker = directory / "round_1/attacker"
        checkpoint = attacker / "checkpoint-300"
        checkpoint.mkdir(parents=True)
        for filename in ("optimizer.pt", "rng_state.pth", "restart_complete.pt", "adapter_model.safetensors"):
            atomic_save({"tensor": torch.tensor([300.0])}, checkpoint / filename)
        atomic_save({"last_epoch": 300, "_last_lr": [5e-5, 5e-5]}, checkpoint / "scheduler.pt")
        races.write_record(checkpoint / "trainer_state.json", {"global_step": 300, "max_steps": 300})
        atomic_save({"config": {**training, "output_dir": str(attacker)},
                     "max_steps": 300, "completed": True}, attacker / "restart_context.pt")
        atomic_save({"config": {"attacker_steps": 300, "plan": {"attacker_steps_per_round": 300}},
                     "phase": "generate_defender", "round": 0, "history": [], "attacker_updates": 300,
                     "metrics": {"before": {"attacker_updates": 0}, "after_attack": {"attacker_updates": 300}}},
                    directory / "race_restart.pt")
        races.write_record(directory / "receipt.json", {"status": "attacker_complete"})
        races.write_record(directory / "reuse_decision.json", {"reuse": False})
        races.write_record(directory / "round_1/before/evaluation.json", {"attacker_updates": 0})
        races.write_record(attacker / "diagnostics/step_000300/metrics.json", {"num_generated": 100})
        manifest["jobs"][name] = {"job_id": "101", "directory": str(directory)}
    path = tmp_path / "source/manifest.json"
    races.write_record(path, manifest)
    monkeypatch.setattr(races, "inputs", lambda: ({}, {}))
    replies = iter(["", "201", "202", "203"])
    monkeypatch.setattr(races.subprocess, "check_output", lambda *args, **kwargs: next(replies))
    return path


def test_extension_preserves_source_native_state_and_defers_defense(tmp_path, monkeypatch):
    source = completed_pilot(tmp_path, monkeypatch)
    hashes = {f: hashlib.sha256(f.read_bytes()).digest() for f in source.parent.rglob("*") if f.is_file()}
    target = races.extend_attacker(source, 600)
    manifest = json.loads(target.read_text())
    assert manifest["protocol"]["attacker_steps_per_round"] == 600
    assert manifest["pause_after_attacker_round"] == 1 and manifest["held"]
    assert manifest["continuation"]["source_steps"] == 300
    for name, job in manifest["jobs"].items():
        directory = target.parent / name
        attacker = directory / "round_1/attacker"
        context = load_checkpoint(attacker / "restart_context.pt")
        assert context["max_steps"] == 600 and not context["completed"]
        assert context["config"]["output_dir"] == str(attacker)
        assert context["config"]["warmup_steps"] == 50
        state = load_checkpoint(directory / "race_restart.pt")
        assert state["phase"] == "attack" and state["round"] == 0 and state["attacker_updates"] == 0
        assert state["config"]["attacker_steps"] == state["config"]["plan"]["attacker_steps_per_round"] == 600
        assert "after_attack" not in state["metrics"]
        assert not (directory / "round_1/after_attack").exists()
        assert (attacker / "diagnostics/step_000300/metrics.json").exists()
        assert "--time=02:00:00" in job["command"] and "--qos=qos_gpu_h100-dev" in job["command"]
        for filename, digest in manifest["continuation"]["checkpoints"][name]["sha256"].items():
            assert hashlib.sha256((attacker / "checkpoint-300" / filename).read_bytes()).hexdigest() == digest
    assert all(hashlib.sha256(f.read_bytes()).digest() == digest for f, digest in hashes.items())


@pytest.mark.parametrize("fault", ["active", "linear", "missing_optimizer", "incomplete", "scheduler",
                                 "group_rate", "inputs"])
def test_extension_rejects_incompatible_sources_before_submitting(tmp_path, monkeypatch, fault):
    source = completed_pilot(tmp_path, monkeypatch)
    manifest = json.loads(source.read_text())
    directory = source.parent / "roberta"
    if fault == "active":
        monkeypatch.setattr(races.subprocess, "check_output", lambda *args, **kwargs: "101")
    elif fault == "linear":
        manifest["protocol"]["execution"]["attacker_training"]["lr_scheduler_type"] = "linear"
        races.write_record(source, manifest)
    elif fault == "missing_optimizer":
        (directory / "round_1/attacker/checkpoint-300/optimizer.pt").unlink()
    elif fault == "incomplete":
        races.write_record(directory / "receipt.json", {"status": "paused"})
    elif fault == "scheduler":
        atomic_save({"last_epoch": 299, "_last_lr": [5e-5]},
                    directory / "round_1/attacker/checkpoint-300/scheduler.pt")
    elif fault == "group_rate":
        atomic_save({"last_epoch": 300, "_last_lr": [5e-5, 1e-5]},
                    directory / "round_1/attacker/checkpoint-300/scheduler.pt")
    else:
        monkeypatch.setattr(races, "inputs", lambda: ({"changed": True}, {}))
    with pytest.raises(ValueError):
        races.extend_attacker(source, 600)
    assert not (tmp_path / "defender_comparison").exists()


def test_native_trainer_extends_completed_constant_schedule_without_rewarming(tmp_path):
    from transformers import Trainer, TrainingArguments, set_seed

    class Regression(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.2)
            self.layer = torch.nn.Linear(2, 1)

        def forward(self, input_ids, labels=None):
            logits = self.layer(self.dropout(input_ids.float())).flatten()
            return {"loss": (logits - labels).square().mean(), "logits": logits}

    dataset = [{"input_ids": torch.tensor([i / 40, 1.0]), "labels": torch.tensor(0.0)} for i in range(40)]

    def make(directory, steps):
        set_seed(42)
        return Trainer(model=Regression(), train_dataset=dataset, args=TrainingArguments(
            output_dir=str(directory), max_steps=steps, per_device_train_batch_size=2,
            learning_rate=5e-5, lr_scheduler_type="constant_with_warmup", warmup_steps=50,
            save_steps=300, save_total_limit=2, report_to="none", use_cpu=True, disable_tqdm=True,
            logging_steps=50,
        ))

    continuous = make(tmp_path / "continuous", 600)
    continuous.train()
    first = make(tmp_path / "extended", 300)
    first.train()
    resumed = make(tmp_path / "extended", 600)
    resumed.train(resume_from_checkpoint=str(tmp_path / "extended/checkpoint-300"))
    assert resumed.state.global_step == 600
    assert resumed.lr_scheduler.last_epoch == 600
    assert all(row["learning_rate"] == 5e-5 for row in resumed.state.log_history
               if row.get("step", 0) > 300 and "learning_rate" in row)
    for name, parameter in resumed.model.named_parameters():
        torch.testing.assert_close(parameter, dict(continuous.model.named_parameters())[name], rtol=0, atol=0)
    assert resumed.optimizer.state_dict()["param_groups"] == continuous.optimizer.state_dict()["param_groups"]
    for key, value in resumed.optimizer.state_dict()["state"].items():
        for field, tensor in value.items():
            torch.testing.assert_close(tensor, continuous.optimizer.state_dict()["state"][key][field], rtol=0, atol=0)
