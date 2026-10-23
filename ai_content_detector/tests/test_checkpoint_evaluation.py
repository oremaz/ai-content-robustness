from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, atomic_save
from ai_content_detector.rl_evasion.config import TextEvasionConfig
from scripts import evaluate_attacker_checkpoint as evaluation


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation.workflow, "OUTPUT", tmp_path)
    root = tmp_path / "rl/encoder"
    path = root / "checkpoint-17"
    path.mkdir(parents=True)
    config = TextEvasionConfig(detector_checkpoint_hash="defender-identity")
    atomic_save({"config": asdict(config), "eval_prompts": ["held-out"]}, root / "restart_context.pt")
    atomic_save({}, path / "restart_complete.pt")
    for name, value in {
        "trainer_state.json": {"global_step": 17, "max_steps": 11988, "epoch": 0.1},
        "adapter_config.json": {},
    }.items():
        (path / name).write_text(json.dumps(value))
    (path / "adapter_model.safetensors").write_bytes(b"fixture adapter")
    for name, value in {"base_pairs.json": {}, "validation_thresholds.json": {},
                        "prompt_manifest.json": {"evaluation": ["held-out"]}}.items():
        (tmp_path / name).write_text(json.dumps(value))
    return path


def test_checkpoint_validation_rejects_uncommitted_or_changed_partition(checkpoint):
    _, metadata = evaluation.checkpoint_metadata("encoder", checkpoint)
    assert metadata["global_step"] == 17
    assert metadata["scheduled_steps"] == 11988
    (evaluation.workflow.OUTPUT / "prompt_manifest.json").write_text('{"evaluation": ["different"]}')
    with pytest.raises(ValueError, match="partition"):
        evaluation.checkpoint_metadata("encoder", checkpoint)
    (checkpoint / "restart_complete.pt").unlink()
    with pytest.raises(ValueError, match="fully committed"):
        evaluation.checkpoint_metadata("encoder", checkpoint)


def test_evaluation_loads_adapter_without_training_and_reuses_completed_result(checkpoint, monkeypatch):
    import peft

    calls = []
    pool = SimpleNamespace(checkpoints={}, clear=lambda: calls.append("clear"))
    monkeypatch.setattr(evaluation.workflow, "load_plan", lambda: {})
    monkeypatch.setattr(evaluation.workflow, "group_pool", lambda *args: pool)
    monkeypatch.setattr(evaluation.workflow, "checkpoint_identity", lambda _: {})
    monkeypatch.setattr(evaluation, "digest", lambda _: "defender-identity")

    class Attacker:
        def __init__(self, config, pool):
            self.config = config
            self.model = SimpleNamespace(requires_grad_=lambda _: None, eval=lambda: None)

        def _setup_model(self):
            calls.append("setup")

        def train(self):
            pytest.fail("Evaluation resumed training")

    monkeypatch.setattr(evaluation.workflow, "ComparisonAttacker", Attacker)
    monkeypatch.setattr(peft, "load_peft_weights", lambda *args, **kwargs: {"saved": 1})

    def restore(model, weights):
        assert weights == {"saved": 1}
        calls.append("restore")
        return SimpleNamespace(unexpected_keys=[], missing_keys=["base.weight"])

    monkeypatch.setattr(peft, "set_peft_model_state_dict", restore)
    monkeypatch.setattr(evaluation.workflow, "evaluate_attacker", lambda *args: {"records": []})
    output = checkpoint.parent / "evaluation"
    result = evaluation.evaluate("encoder", checkpoint, output)
    assert result["checkpoint"]["global_step"] == 17
    assert evaluation.evaluate("encoder", checkpoint, output) == result
    assert calls == ["setup", "restore", "clear", "clear"]
    monkeypatch.setattr(evaluation, "digest", lambda _: "changed-defender")
    with pytest.raises(ValueError, match="defender identity"):
        evaluation.evaluate("encoder", checkpoint, output)


def test_pause_requeues_only_evaluation_job(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")

    def pause(*args, **kwargs):
        raise CheckpointPause("Saved evaluation samples")

    monkeypatch.setattr(evaluation, "evaluate", pause)
    commands = []
    monkeypatch.setattr(evaluation.subprocess, "run", lambda command, **kwargs: commands.append(command))
    evaluation.run("encoder", tmp_path / "checkpoint", tmp_path)
    assert commands == [["scontrol", "requeue", "123"]]
    assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "paused"


def test_scientific_errors_are_not_requeued(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Invalid experiment identity")

    monkeypatch.setattr(evaluation, "evaluate", fail)
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *args, **kwargs: pytest.fail("Retried a scientific error"))
    with pytest.raises(ValueError, match="identity"):
        evaluation.run("encoder", tmp_path / "checkpoint", tmp_path)
    assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "failed"


def test_checkpoint_report_keeps_budgets_and_comparison_explicit(tmp_path, monkeypatch):
    from scripts import plot_research, report_research

    monkeypatch.setattr(report_research, "ROOT", tmp_path)
    monkeypatch.setattr(report_research, "OUTPUT", tmp_path / "results/research_workflow")
    root = report_research.OUTPUT / "defender_comparison"
    directory = root / "checkpoint_evaluations/encoder/step-17"
    (directory / "scores").mkdir(parents=True)
    (root / "warmup").mkdir()
    weights = tmp_path / "checkpoint-17"
    weights.mkdir()
    (weights / "trainer_state.json").write_text('{"log_history": [{"step": 10, "reward": 0.8}]}')
    record = {"detector": "fixture", "attack_success_rate": 0.5}
    (root / "warmup/measurements.json").write_text(json.dumps({"records": [record]}))
    (directory / "scores/measurements.json").write_text(json.dumps({"scores": {"fixture": [0.2, 0.8]}}))
    (directory / "results.json").write_text(json.dumps({
        "checkpoint": {"group": "encoder", "global_step": 17, "scheduled_steps": 11988,
                       "checkpoint": str(weights)}, "records": [record],
        "mean_semantic_similarity": 0.9, "mean_ai_words": 100.0,
    }))
    plots = []
    monkeypatch.setattr(plot_research, "bars", lambda *args, **kwargs: plots.append(args[1]))
    monkeypatch.setattr(plot_research, "lines", lambda *args, **kwargs: plots.append(args[1]))
    text = "\n".join(report_research.checkpoint_evaluations(tmp_path / "docs/research_results.md"))
    assert "update 17" in text and "11988 updates" in text
    assert "matched-budget" in text and "model-selection set" in text
    assert plots == ["paused-encoder-17-evasion", "paused-encoder-17-roc", "paused-encoder-17-reward"]
