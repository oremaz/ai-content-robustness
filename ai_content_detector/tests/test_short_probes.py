import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import short_probes as probes
from scripts.architecture_races import DEFENDERS


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(probes, "ROOT", tmp_path)
    monkeypatch.setattr(probes, "OUTPUT", tmp_path / "results/research_workflow")
    monkeypatch.setattr(probes, "inputs", lambda: ({"identity": "fixed"}, {"seed": 42}))
    monkeypatch.setattr(probes, "split_audit", lambda: {"scored_prompts": 500})
    probes.write_record(probes.OUTPUT / "architecture_races.json", {
        "jobs": {name: {"job_id": str(10 + index)} for index, name in enumerate(DEFENDERS)}})
    calls = []
    monkeypatch.setattr(probes.subprocess, "check_output",
                        lambda command, **kwargs: calls.append(command) or str(100 + len(calls)))
    return probes.submit(), calls


def test_three_development_probes_use_separate_ten_update_attackers(tmp_path, monkeypatch):
    path, calls = fixture(tmp_path, monkeypatch)
    manifest = json.loads(path.read_text())
    assert manifest["attacker_steps"] == 10 and manifest["defender_updates"] == 0
    assert manifest["initialization"] == "base_Qwen"
    assert set(manifest["jobs"]) == set(DEFENDERS)
    assert len({row["directory"] for row in manifest["jobs"].values()}) == 3
    assert all("--qos=qos_gpu_h100-dev" in call and "--time=02:00:00" in call
               and "--requeue" in call and "--hold" in call for call in calls)
    assert Path(manifest["metrics_file"]).exists()
    assert "Unmatched" in manifest["comparison"] and "step-502" in manifest["reference"]


def test_capacity_dependencies_allow_only_one_job_per_branch(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(probes.subprocess, "check_output", lambda *args, **kwargs:
                        "10|PENDING\n11|PENDING\n12|PENDING\n")
    commands = []
    monkeypatch.setattr(probes.subprocess, "run", lambda command, **kwargs: commands.append(command))
    probes.release(path)
    assert commands == [
        ["scontrol", "hold", "10,11,12"],
        ["scontrol", "update", "JobId=10", "Dependency=afterany:101"],
        ["scontrol", "update", "JobId=11", "Dependency=afterany:102"],
        ["scontrol", "update", "JobId=12", "Dependency=afterany:103"],
        ["scontrol", "release", "101,102,103"], ["scontrol", "release", "10,11,12"],
    ]
    assert json.loads(path.read_text())["held"] is False
    assert not any(any("QOS=" in arg or "TimeLimit=" in arg for arg in command) for command in commands)


def test_running_race_is_not_interrupted_to_make_room_for_probe(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(probes.subprocess, "check_output", lambda *args, **kwargs:
                        "10|RUNNING\n11|PENDING\n12|PENDING\n")
    monkeypatch.setattr(probes.subprocess, "run", lambda *args, **kwargs: pytest.fail("Changed running race"))
    with pytest.raises(RuntimeError, match="no longer pending"):
        probes.release(path)


@pytest.mark.parametrize("name", DEFENDERS)
def test_probe_trains_only_attacker_then_reuses_completed_results(tmp_path, monkeypatch, name):
    from ai_content_detector.rl_evasion.config import TextEvasionConfig
    from ai_content_detector.rl_evasion.defender_comparison import workflow, readiness

    path, _ = fixture(tmp_path, monkeypatch)
    manifest = json.loads(path.read_text())
    calls = []
    pool = SimpleNamespace(specifications={DEFENDERS[name]: {}}, clear=lambda: calls.append("clear"))

    class Attacker:
        def __init__(self, directory):
            self.config = TextEvasionConfig(output_dir=str(directory))
            self.training_history = [{"step": 10, "reward": 0.7}]

        def train(self, max_steps):
            assert max_steps == 10
            checkpoint = Path(self.config.output_dir) / "checkpoint-10"
            checkpoint.mkdir(parents=True)
            (checkpoint / "restart_complete.pt").touch()
            probes.write_record(checkpoint / "trainer_state.json", {"global_step": 10, "max_steps": 10})
            calls.append("train10")

    def setup(plan, group, directory, names):
        assert group == name and names == [DEFENDERS[name]]
        return Attacker(directory), pool

    monkeypatch.setattr(workflow, "attacker_setup", setup)
    monkeypatch.setattr(readiness, "require_readiness", lambda *args: None)
    monkeypatch.setattr(workflow, "evaluate_attacker", lambda *args: {
        "records": [{"detector": DEFENDERS[name], "attack_success_rate": 0.8}],
        "mean_semantic_similarity": 0.9})
    result = probes.probe(manifest, name, SimpleNamespace(requested=False))
    assert result["attacker_steps"] == 10 and result["training_targets"] == [DEFENDERS[name]]
    assert result["config"]["num_train_epochs"] == 1
    assert probes.probe(manifest, name, SimpleNamespace(requested=False)) == result
    assert calls == ["train10", "clear"]


def test_checkpoint_pause_requeues_only_probe(tmp_path, monkeypatch):
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
    from scripts import research_workflow, report_research

    path, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "101")
    monkeypatch.setattr(research_workflow, "source_provenance", lambda: {})
    monkeypatch.setattr(report_research, "update_report", lambda: None)

    def pause(*args):
        raise CheckpointPause("Saved evaluation progress")

    monkeypatch.setattr(probes, "probe", pause)
    commands = []
    monkeypatch.setattr(probes.subprocess, "run", lambda command, **kwargs: commands.append(command))
    probes.run(path, "roberta")
    assert commands == [["scontrol", "requeue", "101"]]
    assert json.loads((path.parent / "roberta/receipt.json").read_text())["status"] == "paused"


def test_report_labels_diagnostic_set_and_unmatched_reference(tmp_path, monkeypatch):
    from scripts import report_research, plot_research

    path, _ = fixture(tmp_path, monkeypatch)
    manifest = json.loads(path.read_text())
    monkeypatch.setattr(report_research, "ROOT", tmp_path)
    monkeypatch.setattr(report_research, "OUTPUT", probes.OUTPUT)
    monkeypatch.setattr(plot_research, "lines", lambda *args, **kwargs: None)
    monkeypatch.setattr(plot_research, "bars", lambda *args, **kwargs: None)
    record = {"detector": "roberta", "attack_success_rate": 0.8, "ai_detection_rate": 0.2}
    value = {"records": [record], "scores": {"roberta": [0.2, 0.1]}, "mean_semantic_similarity": 0.88}
    probes.write_record(Path(manifest["reference"]), value)
    probes.write_record(Path(manifest["reference"]).parent / "scores/measurements.json", value)
    probes.write_record(probes.OUTPUT / "defender_comparison/warmup/measurements.json", value)
    directory = Path(manifest["jobs"]["roberta"]["directory"])
    probes.write_record(directory / "results.json", value)
    probes.write_record(directory / "evaluation/scores/measurements.json", value)
    text = "\n".join(report_research.short_probes(tmp_path / "docs/research_results.md"))
    assert "held-out diagnostic" in text and "unmatched" in text
    assert "502" in text and "Individual target" in text and "does not isolate update count" in text


def test_split_audit_rejects_training_prompt_leakage(tmp_path, monkeypatch):
    monkeypatch.setattr(probes, "OUTPUT", tmp_path)
    root = tmp_path / "defender_comparison"
    prompts = [f"reserved {i}" for i in range(500)]
    probes.write_record(root / "prompt_manifest.json", {"train": ["training"], "evaluation": prompts})
    data = {"evaluation": [{"prompt": prompt} for prompt in prompts], "warmup": [{"prompt": "training"}]}
    probes.write_record(root / "base_pairs.json", data)
    assert probes.split_audit()["exact_prompt_overlap_with_fitting"] == 0
    probes.write_record(root / "prompt_manifest.json", {"train": [prompts[0]], "evaluation": prompts})
    with pytest.raises(ValueError, match="overlaps fitting"):
        probes.split_audit()
