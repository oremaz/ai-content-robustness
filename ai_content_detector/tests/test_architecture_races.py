import json
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts import architecture_races as races


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(races, "ROOT", tmp_path)
    monkeypatch.setattr(races, "OUTPUT", tmp_path / "results/research_workflow")
    monkeypatch.setattr(races, "inputs", lambda: ({"identity": "fixed"}, {"seed": 42}))
    monkeypatch.setenv("USER", "fixture-user")
    calls = []
    monkeypatch.setattr(races.subprocess, "check_output",
                        lambda command, **kwargs: calls.append(command) or str(100 + len(calls)))
    path = races.submit()
    return path, calls


def test_three_independent_regular_twenty_hour_jobs(tmp_path, monkeypatch):
    path, calls = fixture(tmp_path, monkeypatch)
    manifest = json.loads(path.read_text())
    assert set(manifest["jobs"]) == {"roberta", "gemma", "diveye"}
    assert manifest["protocol"]["attacker_steps_per_round"] == 300
    assert manifest["protocol"]["rounds"] == 10
    assert manifest["protocol"]["initialization"] == "base_Qwen"
    assert manifest["gpu_limit"] == 3
    assert len(calls) == 3
    for command in calls:
        assert "--qos=qos_gpu_h100-t3" in command and "--time=20:00:00" in command
        assert "--hold" in command and "--requeue" in command and "--signal=USR1@600" in command
        assert not any("dev" in arg or arg.startswith("--dependency=") for arg in command)
    assert len({job["directory"] for job in manifest["jobs"].values()}) == 3


def test_three_independent_fixed_development_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(races, "OUTPUT", tmp_path)
    monkeypatch.setattr(races, "inputs", lambda: ({}, {}))
    commands = []
    monkeypatch.setattr(races.subprocess, "check_output", lambda command, **kwargs:
                        commands.append(command) or str(100 + len(commands)))
    path = races.submit(pilot=True)
    manifest = json.loads(path.read_text())
    assert set(manifest["jobs"]) == {"roberta", "gemma", "diveye"}
    assert manifest["pause_after_round"] == 3 and manifest["protocol"]["rounds"] == 10
    assert manifest["protocol"]["attacker_steps_per_round"] == 300
    assert manifest["protocol"]["execution"] == {"generation_batch_size": 8,
            "batch_semantic_evaluation": True, "reuse_before_evaluation": True,
            "diagnostic_samples": 100, "attacker_training": {"learning_rate": 5e-5, "lr_scheduler_type": "constant_with_warmup", "warmup_steps": 50}}
    for command in commands:
        assert "--qos=qos_gpu_h100-dev" in command and "--time=02:00:00" in command
        assert "--requeue" in command and "--signal=USR1@600" in command
        assert not any(arg.startswith("--dependency=") for arg in command)


def test_attacker_only_submission_does_not_schedule_defender_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(races, "OUTPUT", tmp_path)
    monkeypatch.setattr(races, "inputs", lambda: ({}, {}))
    replies = iter(["101", "102", "103"])
    monkeypatch.setattr(races.subprocess, "check_output", lambda *args, **kwargs: next(replies))
    path = races.submit(pilot=True, attacker_only=True)
    manifest = json.loads(path.read_text())
    assert manifest["pause_after_attacker_round"] == 1
    assert "pause_after_round" not in manifest
    assert manifest["protocol"]["attacker_steps_per_round"] == 300
    assert manifest["time_limit"] == "02:00:00"


def test_continuation_reuses_directories_and_rejects_active_writers(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path, monkeypatch)
    manifest = json.loads(path.read_text())
    manifest["pause_after_round"] = 3
    path.write_text(json.dumps(manifest))
    calls = []
    monkeypatch.setattr(races.subprocess, "check_output", lambda *args, **kwargs: "101\n")
    monkeypatch.setattr(races.subprocess, "run", lambda command, **kwargs: calls.append(command))
    with pytest.raises(ValueError, match="active"):
        races.resume(path, through_round=5)
    assert calls == []
    replies = iter(["", "201", "202", "203"])
    monkeypatch.setattr(races.subprocess, "check_output", lambda *args, **kwargs: next(replies))
    races.resume(path, through_round=5)
    resumed = json.loads(path.read_text())
    assert resumed["pause_after_round"] == 5
    assert {name: job["directory"] for name, job in resumed["jobs"].items()} == {
        name: job["directory"] for name, job in manifest["jobs"].items()}
    assert [job["job_id"] for job in resumed["jobs"].values()] == ["201", "202", "203"]
    assert calls == [["scontrol", "release", "201,202,203"]]


def test_release_cli_calls_the_release_function(tmp_path, monkeypatch):
    import runpy
    import sys
    from scripts import research_chain

    path, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(research_chain, "OUTPUT", races.OUTPUT)
    monkeypatch.setattr(sys, "argv", [races.__file__, "release", str(path)])
    calls = []
    monkeypatch.setattr(races.subprocess, "run", lambda command, **kwargs: calls.append(command))
    runpy.run_path(races.__file__, run_name="__main__")
    assert calls == [["scontrol", "release", "101,102,103"]]
    assert not json.loads(path.read_text())["held"]


@pytest.mark.parametrize("wrong_owner", [False, True])
def test_replacement_cancels_only_verified_superseded_jobs(tmp_path, monkeypatch, wrong_owner):
    path, _ = fixture(tmp_path, monkeypatch)
    old = {f"comparison_{kind}_{group}": str(10 + index + offset)
           for offset, kind in [(0, "rl"), (3, "race")]
           for index, group in enumerate(("encoder", "decoder", "ensemble"))}
    races.write_record(races.OUTPUT / "native_chain.json", {"jobs": {**old, "image_notebook": "99"}})
    owner = "different-user" if wrong_owner else "fixture-user"
    rows = "\n".join(f"{job}|{owner}|research-{stage}|PENDING" for stage, job in old.items())
    monkeypatch.setattr(races.subprocess, "check_output", lambda *args, **kwargs: rows)
    commands = []
    monkeypatch.setattr(races.subprocess, "run", lambda command, **kwargs: commands.append(command))
    if wrong_owner:
        with pytest.raises(ValueError, match="identity"):
            races.replace_and_release(path)
        assert commands == []
    else:
        races.replace_and_release(path)
        assert commands == [["scancel", *old.values()], ["scontrol", "release", "101,102,103"]]
        assert json.loads(path.read_text())["held"] is False
        assert "99" not in commands[0]


def test_submission_failure_cancels_only_new_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(races, "ROOT", tmp_path)
    monkeypatch.setattr(races, "OUTPUT", tmp_path)
    monkeypatch.setattr(races, "inputs", lambda: ({}, {}))
    replies = iter(["101", "invalid"])
    monkeypatch.setattr(races.subprocess, "check_output", lambda *args, **kwargs: next(replies))
    commands = []
    monkeypatch.setattr(races.subprocess, "run", lambda command, **kwargs: commands.append(command))
    with pytest.raises(RuntimeError, match="response"):
        races.submit()
    assert commands == [["scancel", "101"]]


@pytest.mark.parametrize("pause", [True, False])
def test_race_runner_requeues_only_saved_pauses(tmp_path, monkeypatch, pause):
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
    from ai_content_detector.rl_evasion.defender_comparison import workflow, readiness
    from scripts import research_workflow, report_research

    path, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "101")
    monkeypatch.setattr(research_workflow, "source_provenance", lambda: {"source_fingerprint": "fixture"})
    monkeypatch.setattr(report_research, "update_report", lambda: None)
    monkeypatch.setattr(readiness, "require_readiness", lambda *args: None)
    committed = path.parent / "roberta/saved-state"

    def race(plan, group, **kwargs):
        assert group == "roberta" and kwargs["defender_names"] == ["roberta"]
        assert kwargs["attacker_steps"] == 300 and kwargs["rounds"] == 10
        assert kwargs["reuse_standalone"] is False
        if pause:
            committed.write_text("saved optimizer progress")
            raise CheckpointPause("saved")
        raise ValueError("scientific failure")

    monkeypatch.setattr(workflow, "race", race)
    commands = []
    monkeypatch.setattr(races.subprocess, "run", lambda command, **kwargs: commands.append(command))
    if pause:
        races.run(path, "roberta")
        assert committed.read_text() == "saved optimizer progress"
        assert commands == [["scontrol", "requeue", "101"]]
    else:
        with pytest.raises(ValueError, match="scientific failure"):
            races.run(path, "roberta")
        assert commands == []
    receipt = json.loads((path.parent / "roberta/receipt.json").read_text())
    assert receipt["status"] == ("paused" if pause else "failed")


def test_partial_race_report_exports_measured_round_and_training_curves(tmp_path, monkeypatch):
    from scripts import report_research

    output = tmp_path / "results/research_workflow"
    monkeypatch.setattr(report_research, "ROOT", tmp_path)
    monkeypatch.setattr(report_research, "OUTPUT", output)
    directory = output / "defender_comparison/races_100/fixture/roberta"
    directory.mkdir(parents=True)
    job = {"job_id": "101", "directory": str(directory)}
    races.write_record(directory.parent / "manifest.json", {
        "protocol": {"rounds": 10, "attacker_steps_per_round": 100}, "jobs": {"roberta": job}})
    races.write_record(directory / "receipt.json", {"status": "paused"})
    history = {"round": 1}
    for phase, asr in [("before", 0.1), ("after_attack", 0.8), ("after_defense", 0.2)]:
        history[phase] = {"records": [{"detector": "roberta", "attack_success_rate": asr,
                                      "ai_detection_rate": 1 - asr, "human_false_positive_rate": 0.01,
                                      "roc_auc": 0.9, "num_generated": 500, "num_ineligible_pairs": 0}],
                          "mean_semantic_similarity": 0.88}
    races.write_record(directory / "history.json", [history])
    races.write_record(directory / "reuse_decision.json", {"reason": "Base Qwen"})
    races.write_record(directory / "round_1/training_history.json", [{"step": 10, "reward": 0.7, "loss": 0.01, "learning_rate": 5e-5}])
    text = "\n".join(report_research.architecture_races(tmp_path / "docs/research_results.md"))
    assert "completed rounds: 1/10" in text
    assert "0.8000" in text and "semantic_similarity" in text
    prefix = tmp_path / "docs/figures/research/race100-fixture-roberta"
    for name in ("roberta", "reward", "loss", "learning_rate"):
        for extension in ("csv", "png", "svg"):
            assert Path(f"{prefix}-{name}.{extension}").is_file()


def test_shared_metrics_are_concurrent_and_idempotent(tmp_path):
    path = tmp_path / "metrics.csv"
    races.write_metrics(path, [])
    measurement = {"records": [{"detector": "fixture", "attack_success_rate": 0.8,
                                 "ai_detection_rate": 0.2, "human_false_positive_rate": 0.01}],
                   "mean_semantic_similarity": 0.88}

    def record(name):
        for phase in ["before", "after_attack", "after_defense"]:
            races.save_metrics(path, name, 1, phase, measurement, 100)
            races.save_metrics(path, name, 1, phase, measurement, 100)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(record, races.DEFENDERS))
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert {row["race"] for row in rows} == set(races.DEFENDERS)
    for row in rows:
        assert row["attacker_updates"] == ("0" if row["phase"] == "before" else "100")
        assert row["defender_round"] == ("1" if row["phase"] == "after_defense" else "0")
    changed = {**measurement, "mean_semantic_similarity": 0.5}
    with pytest.raises(ValueError, match="Conflicting metrics"):
        races.save_metrics(path, "gemma", 1, "after_attack", changed, 100)
    with path.open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 9


def test_attacker_only_report_has_partial_phase_table_and_diagnostic_curves(tmp_path, monkeypatch):
    from scripts import report_research

    output = tmp_path / "results/research_workflow"
    monkeypatch.setattr(report_research, "ROOT", tmp_path)
    monkeypatch.setattr(report_research, "OUTPUT", output)
    directory = output / "defender_comparison/races_100/fixture/roberta"
    races.write_record(directory.parent / "manifest.json", {
        "protocol": {"rounds": 10, "attacker_steps_per_round": 500},
        "pause_after_attacker_round": 1, "jobs": {"roberta": {"job_id": "101", "directory": str(directory)}}})
    races.write_record(directory / "receipt.json", {"status": "attacker_complete"})
    for step, asr in ((50, 0.0), (100, 0.03125)):
        races.write_record(directory / f"round_1/attacker/diagnostics/step_{step:06d}/metrics.json", {
            "num_generated": 32, "records": [{"detector": "roberta", "attack_success_rate": asr}]})
    races.write_record(directory / "round_1/after_attack/evaluation.json", {
        "num_generated": 500, "records": [{"detector": "roberta", "attack_success_rate": 0.8,
                                           "ai_detection_rate": 0.2, "num_generated": 500}]})
    text = "\n".join(report_research.architecture_races(tmp_path / "docs/research_results.md"))
    assert "completed rounds: 0/10" in text and "attacker_complete" in text
    assert "after_attack" in text and "0.8000" in text
    assert "logging only" in text and "0.0312" in text
    prefix = tmp_path / "docs/figures/research/race100-fixture-roberta-diagnostic-evasion"
    for extension in ("csv", "png", "svg"):
        assert Path(f"{prefix}.{extension}").is_file()
