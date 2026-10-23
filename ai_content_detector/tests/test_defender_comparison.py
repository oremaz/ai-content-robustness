from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, atomic_save
from ai_content_detector.rl_evasion.config import TextEvasionConfig, seed_everything
from ai_content_detector.rl_evasion.defender_comparison.defenders import (
    DefenderPool, DefenderTrainer, FeatureDefender, ScoringClassifier,
)
from ai_content_detector.rl_evasion.defender_comparison.protocol import classification_metrics, validation_threshold
from ai_content_detector.rl_evasion.defender_comparison.workflow import BatchReward, milestone_reuse_decision


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(20, 8)
        self.output = torch.nn.Linear(8, 20)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def test_decoder_validation_accumulates_only_classification_logits(tmp_path):
    from datasets import Dataset
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Trainer, TrainingArguments
    from ml_pipeline.pipelines_torch.models import HuggingFaceQLoRAWrapper

    config = Gemma4Config(text_config=dict(
        vocab_size=24, vocab_size_per_layer_input=24,
        hidden_size=16, hidden_size_per_layer_input=4, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=8, layer_types=["full_attention"],
        rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 10000.0}},
    ))
    model = HuggingFaceQLoRAWrapper._attach_sequence_classification_head(
        Gemma4ForConditionalGeneration(config), 16, 0.02, 2, "classification", torch.float32,
    ).eval()
    data = Dataset.from_dict({"input_ids": [[1, 2, 3, 4]] * 8,
                              "attention_mask": [[1, 1, 1, 1]] * 8, "labels": [0, 1] * 4})
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none",
                             per_device_eval_batch_size=2)
    raw = Trainer(model=model, args=args).predict(data)
    assert isinstance(raw.predictions, tuple)
    actual = DefenderTrainer(model=model, args=args).predict(data)
    assert isinstance(actual.predictions, np.ndarray)
    assert actual.predictions.shape == (8, 2)
    np.testing.assert_array_equal(actual.predictions, raw.predictions[0])
    assert actual.metrics["test_loss"] == raw.metrics["test_loss"]
    np.testing.assert_array_equal(actual.label_ids, raw.label_ids)


@pytest.mark.parametrize("family", ["binoculars", "fast_detect_gpt"])
def test_supervised_statistics_backpropagate_only_through_scorer_and_head(family):
    torch.manual_seed(42)
    model = ScoringClassifier(family, TinyLM(), TinyLM(), threshold=0.9 if family == "binoculars" else 0)
    model.train()
    ids = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])
    result = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=torch.tensor([0, 1]))
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert not model.reference.training
    assert all(parameter.grad is None for parameter in model.reference.parameters())
    assert sum(parameter.grad.abs().sum() for parameter in model.scorer.parameters()) > 0
    assert model.head.weight.grad.abs().sum() > 0


def test_partial_detector_checkpoint_resumes_native_optimizer_and_best_validation_weights(tmp_path, monkeypatch):
    from ai_content_detector.tests.test_defender_restart import make_defender
    from ai_content_detector.rl_evasion.arms_race import radar_defender

    def make(path):
        defender = make_defender(path)
        defender.ai_label = 0
        defender.paired_validation = True
        defender.trainer_class = DefenderTrainer
        return defender

    ai, human = [f"ai{i}" for i in range(8)], [f"human{i}" for i in range(8)]
    seed_everything(42)
    baseline = make(tmp_path / "baseline")
    expected = baseline.retrain(ai, human)
    seed_everything(42)
    interrupted = make(tmp_path / "resumed")
    native_save = radar_defender.atomic_transformers_save

    def interrupt(trainer, method, model, trial=None, extra_state=None):
        native_save(trainer, method, model, trial, extra_state)
        if trainer.state.global_step == 2:
            raise CheckpointPause("simulated native checkpoint interruption")

    monkeypatch.setattr(radar_defender, "atomic_transformers_save", interrupt)
    with pytest.raises(CheckpointPause):
        interrupted.retrain(ai, human)
    monkeypatch.setattr(radar_defender, "atomic_transformers_save", native_save)
    resumed = make(tmp_path / "resumed")
    actual = resumed.retrain(ai, human)
    for name, parameter in baseline.model.state_dict().items():
        assert torch.equal(parameter, resumed.model.state_dict()[name]), name
    assert actual["auroc"] == expected["auroc"]
    assert actual["eval_loss"] == expected["eval_loss"]
    assert list((tmp_path / "resumed").glob("round_1/checkpoint-*/defender_trainable.pt"))


def test_warmup_continues_existing_feature_classifier_and_completed_replay(tmp_path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    ai = [f"generated synthetic systematic prose topic{i}" for i in range(10)]
    human = [f"personal natural handwritten account topic{i}" for i in range(10)]
    vectorizer = TfidfVectorizer().fit(ai + human)
    classifier = LogisticRegression().fit(vectorizer.transform(ai + human), [0] * 10 + [1] * 10)
    defender = FeatureDefender({"family": "tfidf", "model_name": "fixture"}, tmp_path)
    defender.extractor, defender.classifier = vectorizer, classifier
    metrics = defender.retrain(ai, human, warmup=True)
    scores = defender.predict_batch(ai)
    replay = FeatureDefender(defender.specification, tmp_path)
    replay.extractor = vectorizer
    assert replay.retrain(ai, human, warmup=True) == metrics
    assert replay.predict_batch(ai) == scores
    assert metrics["train_samples"] == 16 and metrics["val_samples"] == 4


def test_validation_threshold_handles_ties_without_exceeding_requested_fpr():
    scores = [0.8] * 10 + [0.3] * 90
    threshold = validation_threshold(scores)
    assert np.mean(np.asarray(scores) >= threshold) <= 0.01
    metrics = classification_metrics([0.9, 0.95], [0.1, 0.2])
    assert metrics["roc_auc"] == 1 and metrics["human_false_positive_rate"] == 0


def test_nested_scheduler_requests_reach_both_stage_and_inner_loop():
    import signal
    from ai_content_detector.rl_evasion.checkpointing import StopRequest
    with StopRequest() as outer:
        with StopRequest() as inner:
            signal.raise_signal(signal.SIGUSR1)
            assert inner.requested and outer.requested


def test_batch_reward_queries_each_detector_once_and_rejects_short_outputs():
    calls = []
    pool = SimpleNamespace(specifications={"one": {}, "two": {}},
                           predict_batch=lambda texts: calls.append(texts) or
                           {"one": [0.2] * len(texts), "two": [0.4] * len(texts)})
    reward = BatchReward(pool)
    reward.semantic = lambda source, text: 0.9
    text = " ".join(["word"] * 20)
    scores = reward.batch([text] * 3, [text, "short", text])
    assert len(calls) == 1 and len(calls[0]) == 2
    assert scores[1]["rejected"] and scores[1]["total"] == 0
    assert scores[0]["evasion"] == pytest.approx(0.7)


def test_step_500_reuse_rejects_a_longer_standalone_learning_rate_schedule(tmp_path):
    cfg = TextEvasionConfig()
    milestone = tmp_path / "milestones/step_500"
    milestone.mkdir(parents=True)
    (milestone / "trainer_state.json").write_text(json.dumps({"max_steps": 3000, "global_step": 500}))
    atomic_save({"config": asdict(cfg)}, milestone / "research_restart_context.pt")
    decision = milestone_reuse_decision(tmp_path, cfg, {})
    assert decision["reuse"] is False
    assert "learning_rate_schedule_horizon" in decision["reason"]
    (milestone / "trainer_state.json").write_text(json.dumps({"max_steps": 500, "global_step": 500}))
    assert milestone_reuse_decision(tmp_path, cfg, {})["reuse"] is True


def test_only_the_requested_two_single_defenders_and_full_seven_detector_ensemble():
    from pathlib import Path
    from ai_content_detector.rl_evasion.defender_comparison.plan import GROUPS, specifications
    selection = {"selected": {"tfidf": "tfidf_logreg", "encoder": "qlora_modernbert_large", "decoder": "qlora_gemma"},
                 "checkpoints": {name: {"checkpoint_id": name} for name in ("tfidf", "encoder", "decoder")}}
    pool = specifications(selection, Path("results"))
    assert len(pool) == 7
    assert GROUPS["encoder"] == ["modernbert_large"]
    assert GROUPS["decoder"] == ["decoder_classifier"]
    assert GROUPS["ensemble"] is None


def test_loading_a_defender_does_not_change_the_attackers_sampling_rng(tmp_path, monkeypatch):
    from ai_content_detector.rl_evasion.defender_comparison import defenders

    class RandomLoadingDefender:
        def __init__(self, *args, **kwargs):
            pass

        def setup(self):
            torch.manual_seed(99)
            torch.rand(50)

        def move(self, device):
            pass

    monkeypatch.setattr(defenders, "TorchDefender", RandomLoadingDefender)
    torch.manual_seed(42)
    expected = torch.rand(3)
    torch.manual_seed(42)
    pool = DefenderPool({"fixture": {"family": "notebook_hf"}}, tmp_path, device="cpu")
    pool.get("fixture")
    assert torch.equal(torch.rand(3), expected)


def test_scoring_can_release_cpu_residency_between_defenders(tmp_path, monkeypatch):
    import weakref
    from ai_content_detector.rl_evasion.defender_comparison import defenders
    loads = []
    moves = []

    class Detector:
        def __init__(self, *args, **kwargs):
            pass

        def setup(self):
            loads.append(1)

        def move(self, device):
            moves.append(device)

    monkeypatch.setattr(defenders, "TorchDefender", Detector)
    pool = DefenderPool({"fixture": {"family": "notebook_hf"}}, tmp_path, device="cpu", retain_on_cpu=False)
    detector = pool.get("fixture")
    reference = weakref.ref(detector)
    del detector
    pool.release("fixture")
    assert reference() is None and pool.defenders == {}
    pool.get("fixture")
    assert len(loads) == 2 and moves == ["cpu", "cpu", "cpu"]


@pytest.mark.parametrize("group,names,steps,rounds", [
    ("encoder", None, 500, 10), ("roberta", ["one"], 100, 2),
    ("gemma", ["one"], 100, 2), ("diveye", ["one"], 100, 2),
])
def test_race_resumes_a_committed_phase_without_repeating_attacker_updates(tmp_path, monkeypatch,
                                                                        group, names, steps, rounds):
    from ai_content_detector.rl_evasion.defender_comparison import workflow
    from ai_content_detector.rl_evasion.defender_comparison.protocol import write_json
    monkeypatch.setattr(workflow, "OUTPUT", tmp_path)
    pairs = [{"prompt": "prompt", "reference": "human", "ai": "ai", "human": "human", "eligible": True}]
    write_json(tmp_path / "base_pairs.json", {"warmup": pairs, "evaluation": pairs})
    initial = tmp_path / "warm"
    atomic_save({"model": {}}, initial / "completed.pt")
    updates = []

    class Attacker:
        def __init__(self, directory):
            self.config = TextEvasionConfig(output_dir=str(directory))
            self.model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.model.weight.fill_(0)
            self.tokenizer = None
            self.training_history = []

        def train(self, max_steps=None):
            assert max_steps == steps
            with torch.no_grad():
                self.model.weight.add_(1)
            updates.append(float(self.model.weight.item()))

    class Defender:
        def retrain(self, *args, **kwargs):
            self._round += 1
            atomic_save({"model": {}}, self.output_dir / f"round_{self._round}/completed.pt")

    class Pool:
        def __init__(self):
            self.specifications = {"one": {}}
            self.checkpoints = {"one": str(initial)}

        def get(self, name):
            return Defender()

        def release(self, name):
            pass

        def clear(self):
            pass

    def setup(plan, group, directory, **kwargs):
        assert kwargs.get("names") == names
        return Attacker(directory), Pool()

    monkeypatch.setattr(workflow, "attacker_setup", setup)
    monkeypatch.setattr(workflow, "generation", lambda *args, **kwargs: ["generated"])
    monkeypatch.setattr(workflow, "matched_pairs", lambda *args: pairs)
    monkeypatch.setattr(workflow, "evaluate_attacker", lambda attacker, pool, *args, **kwargs: {
        "records": [{"detector": "one", "attack_success_rate": float(attacker.model.weight.item()) / 10}]})
    save = workflow.atomic_save
    interrupted = False

    def interrupt(state, path):
        nonlocal interrupted
        save(state, path)
        if path.name == "race_restart.pt" and state["round"] == 0 and state["phase"] == "after_attack" and not interrupted:
            interrupted = True
            raise CheckpointPause("simulated committed phase interruption")

    monkeypatch.setattr(workflow, "atomic_save", interrupt)
    with pytest.raises(CheckpointPause):
        workflow.race({}, group, defender_names=names, attacker_steps=steps, rounds=rounds)
    monkeypatch.setattr(workflow, "atomic_save", save)
    published = []
    measurements = []
    result = workflow.race({}, group, defender_names=names, attacker_steps=steps, rounds=rounds,
                           on_round=lambda: published.append(True),
                           on_measurement=lambda index, phase, values: measurements.append((index, phase)))
    assert updates == list(range(1, rounds + 1))
    assert len(result["history"]) == rounds
    assert len(published) == rounds
    assert (1, "after_attack") in measurements and (1, "after_defense") in measurements
    assert len(list((tmp_path / "race" / group).glob("round_*/*/evaluation.json"))) == rounds * 3
    for row in result["history"]:
        assert row["after_attack"] == row["after_defense"]
    with pytest.raises(ValueError, match="configuration differs"):
        workflow.race({}, group, defender_names=names, attacker_steps=steps + 1, rounds=rounds)


def test_comparison_report_exports_real_metric_tables_and_roc_source_files(tmp_path):
    from scripts.report_defender_comparison import render_stage
    from ai_content_detector.rl_evasion.defender_comparison.protocol import write_json
    directory = tmp_path / "results/research_workflow/defender_comparison"
    record = {"detector": "fixture", **classification_metrics([0.8, 0.9], [0.1, 0.2]),
              "num_ineligible_pairs": 0}
    write_json(directory / "comparison_baseline_measurements.json", {"records": [record]})
    write_json(directory / "baseline/measurements.json", {"scores": {"fixture": [0.8, 0.9, 0.1, 0.2]}})
    rendered = render_stage(tmp_path, "comparison_baseline")
    assert "fixture" in rendered and "1.0000" in rendered
    assert "comparison_baseline-roc.png" in rendered
    assert (tmp_path / "docs/figures/research/comparison_baseline-roc.csv").exists()
    assert (tmp_path / "docs/figures/research/comparison_baseline-detection.svg").exists()


def test_readiness_uses_validation_and_respects_weak_ensemble_member_policy(tmp_path):
    from ai_content_detector.rl_evasion.defender_comparison.readiness import require_readiness
    from ai_content_detector.rl_evasion.defender_comparison.protocol import write_json
    with pytest.raises(RuntimeError, match="user's minimum"):
        require_readiness(tmp_path, None)
    write_json(tmp_path / "validation_scores.json", {"strong": [0.9] * 20 + [0.1] * 20,
                                                    "weak": [0.2] * 20 + [0.3] * 20})
    target = {"minimum_tpr": 0.9, "maximum_fpr": 0.05, "block_ensemble_on_weak_member": False}
    write_json(tmp_path / "readiness_target.json", target)
    assert require_readiness(tmp_path, None)["ensemble"]["ready"]
    assert require_readiness(tmp_path, ["strong"])["strong"]["ready"]
    target["block_ensemble_on_weak_member"] = True
    write_json(tmp_path / "readiness_target.json", target)
    with pytest.raises(RuntimeError, match="weak"):
        require_readiness(tmp_path, None)


def test_controller_adopts_current_job_before_rechecking_completed_notebooks(monkeypatch):
    from scripts import submit_research
    states = iter(["RUNNING|\n", "COMPLETING|\n", "COMPLETED|\n"])
    sleeps = []
    monkeypatch.setattr(submit_research.subprocess, "check_output", lambda *args, **kwargs: next(states))
    monkeypatch.setattr(submit_research.time, "sleep", lambda duration: sleeps.append(duration))
    submit_research.wait_for_existing_job(42)
    assert sleeps == [15, 15]


def test_readiness_report_only_does_not_require_unapproved_quality_targets(tmp_path):
    from ai_content_detector.rl_evasion.defender_comparison.readiness import require_readiness
    from ai_content_detector.rl_evasion.defender_comparison.protocol import write_json
    write_json(tmp_path / "readiness_target.json", {"mode": "report_only"})
    write_json(tmp_path / "validation_scores.json", {"weak": [0.1] * 20 + [0.9] * 20})
    report = require_readiness(tmp_path, None)
    assert report["weak"] == {"tpr": 0.0, "fpr": 1.0, "threshold": 0.5, "ready": None}
    assert not json.loads((tmp_path / "readiness_report.json").read_text())["quality_gate_enforced"]


def test_text_comparison_does_not_wait_for_image_results_or_rewrite_its_plan(tmp_path, monkeypatch):
    import hashlib
    from ai_content_detector.rl_evasion.defender_comparison import workflow
    from ai_content_detector.rl_evasion.defender_comparison.protocol import write_json
    output = tmp_path / "results/research_workflow/defender_comparison"
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "OUTPUT", output)
    write_json(output.parent / "text_notebook.json", {"status": "completed"})
    write_json(output.parent / "text_notebook_measurements.json", {"fixture": "completed text results"})
    selection = {"selected": {"tfidf": "tfidf_lightgbm", "encoder": "qlora_modernbert_large",
                               "decoder": "qlora_gemma4_e4b"},
                 "checkpoints": {name: {"checkpoint_id": name} for name in ("tfidf", "encoder", "decoder")},
                 "measurements_sha256": hashlib.sha256(
                     (output.parent / "text_notebook_measurements.json").read_bytes()).hexdigest()}
    write_json(output / "notebook_selection.json", selection)
    plan = workflow.load_plan()
    plan_path = output / "plan.json"
    modified = plan_path.stat().st_mtime_ns
    assert workflow.load_plan() == plan
    assert plan_path.stat().st_mtime_ns == modified
    assert not (output.parent / "image_notebook.json").exists()


def test_notebook_inference_adapter_is_reenabled_for_supervised_warmup(tmp_path, monkeypatch):
    from peft import LoraConfig, get_peft_model
    from transformers import RobertaConfig, RobertaForSequenceClassification
    from ai_content_detector.detectors import text_detectors
    from ai_content_detector.rl_evasion.defender_comparison.defenders import TorchDefender
    from ml_pipeline.utils import utils
    model = get_peft_model(RobertaForSequenceClassification(RobertaConfig(
        vocab_size=24, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=32, num_labels=2,
    )), LoraConfig(r=2, lora_alpha=4, target_modules=["query", "value"], task_type="SEQ_CLS"))
    model.requires_grad_(False)

    class Detector:
        def __init__(self, **kwargs):
            self._available = True
            self._model = SimpleNamespace(model=model, tokenizer=SimpleNamespace(pad_token="pad"))

        def _load(self):
            pass

    monkeypatch.setattr(text_detectors, "ModernBERTDetector", Detector)
    monkeypatch.setattr(utils, "get_checkpoint_id", lambda *args: "checkpoint")
    defender = TorchDefender({"family": "notebook_hf", "model_name": "fixture", "checkpoint_id": "checkpoint"},
                             tmp_path, device="cpu")
    defender.setup()
    result = defender.model(input_ids=torch.tensor([[1, 3, 4, 2], [1, 5, 6, 2]]), labels=torch.tensor([0, 1]))
    result.loss.backward()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert any("lora_B" in name and grad is not None and grad.abs().sum() > 0 for name, grad in gradients.items())
    assert any("classifier" in name and grad is not None and grad.abs().sum() > 0 for name, grad in gradients.items())
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
