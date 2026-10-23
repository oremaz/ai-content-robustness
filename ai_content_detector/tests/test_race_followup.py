import copy
import json
from pathlib import Path

import pytest
import torch

from ai_content_detector.rl_evasion.checkpointing import atomic_save, load_checkpoint
from ai_content_detector.rl_evasion.config import TextEvasionConfig
from ai_content_detector.rl_evasion.text_evasion.schedules import configure_scheduler, warmup_stable_cosine
from scripts import architecture_races as races


def test_schedule_preserves_all_prior_rates_and_resumes_during_cosine_decay():
    from transformers import get_scheduler

    def optimizer():
        return torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=5e-5)

    original = optimizer()
    constant = get_scheduler("constant_with_warmup", original, num_warmup_steps=50, num_training_steps=600)
    target = optimizer()
    cosine = warmup_stable_cosine(target, num_training_steps=750, num_warmup_steps=50,
                                  num_decay_steps=100, min_lr_ratio=0.2)
    for step in range(601):
        assert constant.get_last_lr() == cosine.get_last_lr()
        if step < 600:
            original.step(); constant.step()
            target.step(); cosine.step()
    resumed = optimizer()
    resumed_cosine = warmup_stable_cosine(resumed, num_training_steps=750, num_warmup_steps=50,
                                          num_decay_steps=100, min_lr_ratio=0.2)
    resumed.load_state_dict(original.state_dict())
    resumed_cosine.load_state_dict(constant.state_dict())
    rates = {600: resumed_cosine.get_last_lr()[0]}
    saved = None
    for step in range(601, 751):
        resumed.step(); resumed_cosine.step()
        rates[step] = resumed_cosine.get_last_lr()[0]
        if step == 700:
            saved = copy.deepcopy((resumed.state_dict(), resumed_cosine.state_dict()))
    assert rates[650] == 5e-5
    assert rates[675] == pytest.approx(1e-5 + 4e-5 * (1 + 2 ** -0.5) / 2)
    assert rates[700] == pytest.approx(3e-5)
    assert rates[750] == pytest.approx(1e-5)
    assert all(rates[step] >= rates[step + 1] for step in range(650, 750))
    restarted = optimizer()
    restarted_cosine = warmup_stable_cosine(restarted, num_training_steps=750, num_warmup_steps=50,
                                            num_decay_steps=100, min_lr_ratio=0.2)
    restarted.load_state_dict(saved[0]); restarted_cosine.load_state_dict(saved[1])
    for step in range(701, 751):
        restarted.step(); restarted_cosine.step()
        assert restarted_cosine.get_last_lr()[0] == rates[step]


def source_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(races, "ROOT", tmp_path)
    monkeypatch.setattr(races, "OUTPUT", tmp_path / "results/research_workflow")
    monkeypatch.setenv("USER", "fixture-user")
    monkeypatch.setattr(races, "inputs", lambda: ({}, {}))
    manifest = {"protocol": {"rounds": 10, "attacker_steps_per_round": 600, "defenders": races.DEFENDERS,
                           "execution": {"diagnostic_samples": 100, "attacker_training": {
                               "learning_rate": 5e-5, "lr_scheduler_type": "constant_with_warmup", "warmup_steps": 50}}},
                "inputs": {}, "baseline_plan": {}, "pause_after_attacker_round": 1,
                "jobs": {name: {"job_id": str(101+i), "directory": str(tmp_path / "source" / name)}
                         for i, name in enumerate(races.DEFENDERS)}}
    path = tmp_path / "source/manifest.json"
    races.write_record(path, manifest)
    replies = iter(["101\n102\n103", "201", "202", "203"])
    commands = []
    monkeypatch.setattr(races.subprocess, "check_output", lambda command, **kwargs:
                        commands.append(command) or next(replies))
    return path, manifest, commands


def test_followup_submits_three_independent_dependencies_and_regular_allocations(tmp_path, monkeypatch):
    source, original, commands = source_fixture(tmp_path, monkeypatch)
    target = races.submit_followup(source)
    manifest = json.loads(target.read_text())
    assert manifest["protocol"]["rounds"] == 3 and manifest["protocol"]["attacker_steps_per_round"] == 750
    assert manifest["qos"] == "qos_gpu_h100-t3" and manifest["time_limit"] == "11:45:00"
    assert manifest["gpu_limit"] == 3 and manifest["max_restarts"] == 100
    assert "pause_after_attacker_round" not in manifest
    for name, job in manifest["jobs"].items():
        assert f"--dependency=afterok:{original['jobs'][name]['job_id']}" in job["command"]
        assert "--kill-on-invalid-dep=yes" in job["command"]
        assert "--time=11:45:00" in job["command"] and "--qos=qos_gpu_h100-t3" in job["command"]
        assert "--requeue" in job["command"] and "--signal=USR1@600" in job["command"]
    assert len(commands) == 4 and json.loads(source.read_text()) == original


def test_followup_copies_only_after_source_completion_and_prepares_once(tmp_path, monkeypatch):
    source, original, _ = source_fixture(tmp_path, monkeypatch)
    target = races.submit_followup(source)
    manifest = json.loads(target.read_text())
    name = "roberta"
    directory = Path(manifest["jobs"][name]["directory"])
    source_dir = Path(original["jobs"][name]["directory"])
    races.write_record(source_dir / "receipt.json", {"status": "running"})
    with pytest.raises((FileNotFoundError, ValueError)):
        races.prepare_followup(manifest, name)
    assert not (directory / "race_restart.pt").exists()
    attacker = source_dir / "round_1/attacker"
    checkpoint = attacker / "checkpoint-600"
    checkpoint.mkdir(parents=True)
    for filename in ("optimizer.pt", "rng_state.pth", "restart_complete.pt", "adapter_model.safetensors"):
        atomic_save({"value": torch.tensor([600])}, checkpoint / filename)
    atomic_save({"last_epoch": 600, "_last_lr": [5e-5, 5e-5]}, checkpoint / "scheduler.pt")
    races.write_record(checkpoint / "trainer_state.json", {"global_step": 600, "max_steps": 600})
    atomic_save({"config": {**original["protocol"]["execution"]["attacker_training"], "output_dir": str(attacker)},
                 "max_steps": 600, "completed": True}, attacker / "restart_context.pt")
    atomic_save({"config": {"plan": {}, "rounds": 10}, "round": 0, "phase": "generate_defender",
                 "history": [], "attacker_updates": 600, "metrics": {"before": {}, "after_attack": {}}},
                source_dir / "race_restart.pt")
    races.write_record(source_dir / "round_1/before/evaluation.json", {"attacker_updates": 0})
    races.write_record(attacker / "diagnostics/step_000600/metrics.json", {"num_generated": 100})
    races.write_record(source_dir / "reuse_decision.json", {"reuse": False})
    races.write_record(source_dir / "receipt.json", {"status": "attacker_complete"})
    races.write_record(directory / ".continuation-staging/interrupted.json", {})
    races.write_record(directory / "round_1/partial.json", {})
    races.prepare_followup(manifest, name)
    state = load_checkpoint(directory / "race_restart.pt")
    context = load_checkpoint(directory / "round_1/attacker/restart_context.pt")
    assert state["phase"] == "attack" and state["attacker_updates"] == 0
    assert state["config"]["rounds"] == 3 and state["config"]["attacker_steps"] == 750
    assert state["config"]["plan"]["execution"] == manifest["protocol"]["execution"]
    assert context["max_steps"] == 750 and not context["completed"]
    assert context["config"]["output_dir"] == str(directory / "round_1/attacker")
    assert context["config"]["lr_scheduler_kwargs"] == {"num_decay_steps": 100, "min_lr_ratio": 0.2}
    assert (directory / "continuation.json").exists() and not (directory / ".continuation-staging").exists()
    assert not (directory / "round_1/partial.json").exists()
    atomic_save({"phase": "defend", "attacker_updates": 750}, directory / "race_restart.pt")
    races.prepare_followup(manifest, name)
    assert load_checkpoint(directory / "race_restart.pt")["phase"] == "defend"
    assert load_checkpoint(attacker / "restart_context.pt")["max_steps"] == 600


def test_native_trainer_migrates_constant_checkpoint_and_resumes_cosine_exactly(tmp_path):
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    class Regression(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.2)
            self.layer = torch.nn.Linear(2, 1)

        def forward(self, input_ids, labels=None):
            logits = self.layer(self.dropout(input_ids.float())).flatten()
            return {"loss": (logits - labels).square().mean(), "logits": logits}

    dataset = [{"input_ids": torch.tensor([i / 40, 1.0]), "labels": torch.tensor(0.0)} for i in range(40)]

    def make(directory, *, constant=False, pause=False):
        set_seed(42)
        cfg = TextEvasionConfig(learning_rate=5e-5, warmup_steps=50,
                               lr_scheduler_type="constant_with_warmup" if constant else "warmup_stable_decay",
                               lr_scheduler_kwargs={} if constant else {"num_decay_steps": 100, "min_lr_ratio": 0.2})
        trainer = Trainer(model=Regression(), train_dataset=dataset, args=TrainingArguments(
            output_dir=str(directory), max_steps=600 if constant else 750, per_device_train_batch_size=2,
            learning_rate=cfg.learning_rate, lr_scheduler_type=cfg.lr_scheduler_type,
            lr_scheduler_kwargs=cfg.lr_scheduler_kwargs, warmup_steps=50,
            save_steps=50, save_total_limit=2, report_to="none", use_cpu=True, disable_tqdm=True,
            logging_steps=50,
        ))
        configure_scheduler(trainer, cfg)

        class Pause(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if pause and state.global_step == 700:
                    control.should_training_stop = True
                return control

        trainer.add_callback(Pause())
        return trainer

    continuous = make(tmp_path / "full")
    continuous.train()
    source = make(tmp_path / "source", constant=True)
    source.train()
    migrated = make(tmp_path / "continued", pause=True)
    migrated.train(resume_from_checkpoint=str(tmp_path / "source/checkpoint-600"))
    assert migrated.state.global_step == 700 and migrated.lr_scheduler.get_last_lr()[0] == pytest.approx(3e-5)
    resumed = make(tmp_path / "continued")
    resumed.train(resume_from_checkpoint=str(tmp_path / "continued/checkpoint-700"))
    assert resumed.state.global_step == 750 and resumed.lr_scheduler.get_last_lr()[0] == pytest.approx(1e-5)
    for name, parameter in resumed.model.named_parameters():
        torch.testing.assert_close(parameter, dict(continuous.model.named_parameters())[name], rtol=0, atol=0)
    for key, value in resumed.optimizer.state_dict()["state"].items():
        for field, tensor in value.items():
            torch.testing.assert_close(tensor, continuous.optimizer.state_dict()["state"][key][field], rtol=0, atol=0)
