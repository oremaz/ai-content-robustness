import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
from ai_content_detector.rl_evasion.config import TextEvasionConfig
from ai_content_detector.rl_evasion.defender_comparison import warm_start


def bundle(directory):
    directory.mkdir(parents=True)
    checkpoint = directory / "checkpoint-1213"
    checkpoint.mkdir()
    paths = {}
    for name, value in {"adapter_model.safetensors": "fixture", "adapter_config.json": "{}",
                        "trainer_state.json": '{"global_step": 1213}'}.items():
        (checkpoint / name).write_text(value)
    values = {"pairs": [{"prompt": "held out", "ai": "saved attack", "human": "human"}],
              "before": {"records": [{"detector": "roberta", "attack_success_rate": 0.0}]},
              "after_attack": {"records": [{"detector": "roberta", "attack_success_rate": 0.8}]}}
    for name, value in values.items():
        path = directory / f"{name}.json"
        path.write_text(json.dumps(value))
        paths[name] = str(path)
    files = list(checkpoint.iterdir()) + [Path(path) for path in paths.values()]
    return {"checkpoint": str(checkpoint), "updates": 1213, **paths,
            "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}


def test_warm_start_rejects_changed_weights_and_incorrect_update_count(tmp_path):
    source = bundle(tmp_path / "source")
    warm_start.verify(source)
    source["updates"] = 500
    with pytest.raises(ValueError, match="update count"):
        warm_start.verify(source)
    source["updates"] = 1213
    Path(source["checkpoint"], "adapter_model.safetensors").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        warm_start.verify(source)


@pytest.mark.parametrize("interrupt_phase", (None, "generate_defender", "defend"))
@pytest.mark.parametrize("defender_only", (False, True))
def test_retained_attacker_begins_with_defense_and_resumes_without_replaying(tmp_path, monkeypatch, interrupt_phase, defender_only):
    from ai_content_detector.rl_evasion.defender_comparison import workflow

    monkeypatch.setattr(workflow, "OUTPUT", tmp_path)
    base = {"evaluation": [{"prompt": "held out"}],
            "warmup": [{"prompt": str(i), "reference": "human " * 24} for i in range(6)]}
    (tmp_path / "base_pairs.json").write_text(json.dumps(base))
    warm = tmp_path / "warm"
    warm.mkdir(); torch.save({"round": 0}, warm / "completed.pt")
    source = bundle(tmp_path / "source")
    trained, defended, loaded, measured = [], [], [], []

    def load(actor, spec):
        loaded.append(spec["updates"])
        with torch.no_grad():
            actor.model.weight.fill_(0.4)

    monkeypatch.setattr(warm_start, "load_policy", load)

    def setup(plan, group, directory, names):
        model = torch.nn.Linear(1, 1, bias=False)
        actor = SimpleNamespace(model=model, tokenizer=None, config=TextEvasionConfig())

        def train(max_steps):
            assert max_steps == 500
            trained.append(Path(actor.config.output_dir).parent.name)
            with torch.no_grad():
                model.weight.add_(0.5)
            actor.training_history = [{"step": max_steps}]

        actor.train = train
        defender = SimpleNamespace(_round=0)

        def retrain(ai, human):
            defender._round += 1
            defended.append(defender._round)
            destination = defender.output_dir / f"round_{defender._round}"
            destination.mkdir(parents=True)
            torch.save({"round": defender._round}, destination / "completed.pt")

        defender.retrain = retrain
        pool = SimpleNamespace(checkpoints={"roberta": str(warm)}, specifications={"roberta": {}},
                               get=lambda name: defender, release=lambda name: None, clear=lambda: None)
        return actor, pool

    def measure(actor, pool, directory, context, **kwargs):
        measured.append(context)
        if "round_1" in str(directory):
            assert kwargs["evaluation_pairs"] == warm_start.read(source, "pairs")
        else:
            assert "evaluation_pairs" not in kwargs
        return {"records": [{"detector": "roberta", "attack_success_rate": float(actor.model.weight.item() / 2)}]}

    monkeypatch.setattr(workflow, "attacker_setup", setup)
    monkeypatch.setattr(workflow, "evaluate_attacker", measure)
    monkeypatch.setattr(workflow, "generation", lambda actor, prompts, *args, **kwargs: ["ai " * 24] * len(prompts))
    monkeypatch.setattr(workflow, "matched_pairs", lambda prompts, refs, outputs, tokenizer:
                        [{"ai": text, "human": ref, "eligible": True} for text, ref in zip(outputs, refs)])
    native_save = workflow.atomic_save
    interrupted = False

    def save(state, path):
        nonlocal interrupted
        native_save(state, path)
        if state["round"] == 0 and state["phase"] == interrupt_phase and not interrupted:
            if interrupt_phase != "defend" or state["checkpoints"]["roberta"] != str(warm):
                interrupted = True
                raise CheckpointPause("fixture interruption")

    monkeypatch.setattr(workflow, "atomic_save", save)
    options = dict(directory=tmp_path / "race", defender_names=["roberta"], rounds=1 if defender_only else 2,
                   attacker_steps=0 if defender_only else 500, reuse_standalone=False, warm_start=source)
    plan = {"execution": {"reuse_before_evaluation": True, "diagnostic_samples": 100}}
    if interrupt_phase:
        with pytest.raises(CheckpointPause):
            workflow.race(plan, "roberta", **options)
    result = workflow.race(plan, "roberta", **options)
    assert loaded == [1213]
    assert trained == ([] if defender_only else ["round_2"])
    assert defended == ([1] if defender_only else [1, 2])
    assert len(measured) == (1 if defender_only else 3)
    assert len(result["history"]) == (1 if defender_only else 2)
    assert result["history"][0]["before"]["attacker_updates"] == 0
    assert result["history"][0]["after_attack"]["attacker_updates"] == 1213
    if not defender_only:
        assert result["history"][1]["after_attack"]["attacker_updates"] == 1713
        assert result["history"][1]["before"] == result["history"][0]["after_defense"]
    repeated = workflow.race(plan, "roberta", **options)
    assert repeated == result
    assert trained == ([] if defender_only else ["round_2"])
    assert defended == ([1] if defender_only else [1, 2])


def test_cached_measurement_uses_mean_scores_and_rejects_misaligned_pairs():
    from scripts.retained_races import cached_measurement

    pairs = [{"sample_id": "one", "ai": "text", "human": "human", "eligible": True},
             {"sample_id": "two", "ai": "text", "human": "human", "eligible": True}]
    artifact = {"eligible_ids": ["one", "two"], "scores": {
        "a": [0.1, 0.9, 0.1, 0.2], "b": [0.7, 0.3, 0.1, 0.2]}}
    measurement = cached_measurement(pairs, artifact, ["a", "b"], {}, 0.9)
    assert measurement["records"][-1]["detector"] == "ensemble_mean"
    assert measurement["records"][-1]["attack_success_rate"] == 0.5
    assert measurement["mean_semantic_similarity"] == 0.9
    with pytest.raises(ValueError, match="do not match"):
        cached_measurement(pairs[::-1], artifact, ["a"], {})


def test_saved_evaluation_pairs_avoid_generation_and_reject_changed_prompts(tmp_path, monkeypatch):
    from ai_content_detector.rl_evasion.defender_comparison import workflow

    monkeypatch.setattr(workflow, "OUTPUT", tmp_path)
    original = {"prompt": "reserved", "sample_id": "reserved", "reference": "human " * 30,
                "human": "human " * 25, "ai": "base " * 25, "eligible": True}
    (tmp_path / "base_pairs.json").write_text(json.dumps({"evaluation": [original]}))
    (tmp_path / "validation_thresholds.json").write_text("{}")
    pairs = [{**original, "ai": "retained " * 25}]
    monkeypatch.setattr(workflow, "generation", lambda *a, **k: pytest.fail("Saved attack texts must be reused"))
    monkeypatch.setattr(workflow, "evaluate_pool", lambda *a, **k: [])
    monkeypatch.setattr(workflow, "score_checkpointed", lambda *a, **k: {"roberta": [0.8, 0.1]})
    monkeypatch.setattr(workflow, "SemanticSimilarityReward", lambda **kwargs: lambda *args: 0.9)
    measured = workflow.evaluate_attacker(SimpleNamespace(), None, tmp_path / "evaluation", "fixture",
                                         evaluation_pairs=pairs)
    assert measured["records"][0]["ai_detection_rate"] == 1.0
    with pytest.raises(ValueError, match="reserved evaluation partition"):
        workflow.evaluate_attacker(SimpleNamespace(), None, tmp_path / "changed", "fixture",
                                   evaluation_pairs=[{**pairs[0], "prompt": "training prompt"}])


def test_curated_report_excludes_legacy_receipts_and_unselected_manifests(tmp_path, monkeypatch):
    from scripts import report_research

    output = tmp_path / "results/research_workflow"
    output.mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/research_results.md").write_text("Discarded trial report")
    old = output / "defender_comparison/races_100/discarded"
    old.mkdir(parents=True)
    (old / "manifest.json").write_text("deliberately invalid unselected manifest")
    (output / "report_selection.json").write_text(json.dumps({"retained": [], "race_manifests": []}))
    monkeypatch.setattr(report_research, "ROOT", tmp_path)
    monkeypatch.setattr(report_research, "OUTPUT", output)
    report_research.update_report()
    text = (tmp_path / "docs/research_results.md").read_text()
    assert "Retained attacker results" in text and "Discarded trial report" not in text
    assert "one defender update per retained attacker" in text
