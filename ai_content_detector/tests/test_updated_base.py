import json
import pytest

from scripts import evaluate_updated_base as base


def test_saved_base_outputs_use_updated_checkpoint_and_resume(tmp_path, monkeypatch):
    from ai_content_detector.rl_evasion.defender_comparison import workflow, protocol

    pairs = [{"id": "reserved-base-output", "ai": "saved base text", "human": "saved human text"}]
    (tmp_path / "base_pairs.json").write_text(json.dumps({"evaluation": pairs}))
    (tmp_path / "validation_thresholds.json").write_text("{}")
    monkeypatch.setattr(workflow, "OUTPUT", tmp_path)
    monkeypatch.setattr(base, "sha256", lambda path: "unchanged")
    monkeypatch.setattr(workflow, "checkpoint_identity", lambda checkpoints: {"fixed": checkpoints})
    checkpoints = {"decoder_classifier": "/updated/round_1"}
    manifest = {"branches": {"gemma": {"checkpoints": checkpoints, "identity": {"fixed": checkpoints}}},
                "input_sha256": {"base_pairs.json": "unchanged"}, "baseline_plan": {},
                "protocol": {"generated_texts": 1, "defenders": {"gemma": ["decoder_classifier"]}}}
    events = []

    class Pool:
        def clear(self):
            events.append("cleared")

    def pool(plan, name, root, **kwargs):
        assert kwargs == {"checkpoints": checkpoints, "names": ["decoder_classifier"]}
        events.append("loaded updated defender")
        return Pool()

    def score(pool, actual_pairs, root, thresholds):
        assert actual_pairs == pairs
        root.mkdir(parents=True)
        (root / "measurements.json").write_text("{}")
        events.append("scored saved outputs")

    monkeypatch.setattr(workflow, "group_pool", pool)
    monkeypatch.setattr(protocol, "evaluate_pool", score)
    monkeypatch.setattr(base, "cached_measurement", lambda *args: {"records": []})
    result = base.evaluate(manifest, "gemma", tmp_path / "result")
    assert result["attacker_updates"] == 0 and result["adaptation"] == "none"
    assert events == ["loaded updated defender", "scored saved outputs", "cleared"]
    assert base.evaluate(manifest, "gemma", tmp_path / "result") == result
    assert len(events) == 3
    manifest["input_sha256"]["base_pairs.json"] = "changed"
    with pytest.raises(ValueError, match="dataset changed"):
        base.evaluate(manifest, "gemma", tmp_path / "result")
    manifest["input_sha256"]["base_pairs.json"] = "unchanged"
    manifest["branches"]["gemma"]["identity"] = {}
    with pytest.raises(ValueError, match="defender changed"):
        base.evaluate(manifest, "gemma", tmp_path / "result")


def test_evaluation_submission_routes_without_attacker_job(tmp_path, monkeypatch):
    races = base.architecture_races
    commands = []
    monkeypatch.setattr(races.subprocess, "check_output", lambda command, **kwargs:
                        commands.append(command) or str(101 + len(commands)))
    manifest = {"protocol": {"defenders": {"gemma": ["decoder_classifier"]}, "attacker_steps_per_round": 0},
                "qos": "qos_gpu_h100-dev", "time_limit": "02:00:00", "jobs": {},
                "job_prefix": "base-retest", "runner_mode": "--updated-base"}
    path = tmp_path / "manifest.json"
    races.submit_jobs(path, manifest)
    command = commands[0]
    assert command[-3:] == ["--updated-base", str(path), "gemma"]
    assert "--job-name=base-retest-gemma" in command
    assert "--hold" in command and "--signal=USR1@600" in command
    assert "--qos=qos_gpu_h100-dev" in command


def test_curated_report_compares_base_detection_separately(tmp_path, monkeypatch):
    from scripts import report_retained_research as report

    (tmp_path / "docs").mkdir()
    result = tmp_path / "base-job"
    result.mkdir()
    record = {"detector": "ensemble_mean", "num_ai": 500, "ai_detection_rate": 0.98,
              "human_false_positive_rate": 0.01, "roc_auc": 0.99}
    (result / "evaluation.json").write_text(json.dumps({"records": [record]}))
    before = {**record, "ai_detection_rate": 0.96}
    manifest = tmp_path / "manifest.json"
    evaluation = tmp_path / "base-manifest.json"
    evaluation.write_text(json.dumps({"branches": {"ensemble": {"before": {"records": [before]}}},
                                      "jobs": {"ensemble": {"directory": str(result), "job_id": "123"}},
                                      "metrics_file": str(tmp_path / "metrics.csv")}))
    manifest.write_text(json.dumps({"jobs": {}, "base_evaluation_manifest": str(evaluation)}))
    monkeypatch.setattr(report, "architecture_races", lambda *args, **kwargs: [])
    monkeypatch.setattr(report, "bars", lambda *args, **kwargs: None)
    output = report.update_retained_report(tmp_path, tmp_path, {"race_manifests": [str(manifest)], "retained": []})
    text = output.read_text()
    assert "Updated defenders on the base model" in text
    assert "original 500 reserved base-Qwen outputs" in text
    assert "0.9600" in text and "0.9800" in text
    assert "../metrics.csv" in text
