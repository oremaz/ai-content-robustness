from __future__ import annotations

from types import SimpleNamespace, MethodType

import numpy as np
import pytest
import torch

from ai_content_detector.rl_evasion.arms_race import meta_adapt
from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, load_checkpoint
from ai_content_detector.rl_evasion.config import seed_everything


def test_maximum_detector_pool_requires_real_scores_and_tracks_missing_resources(monkeypatch):
    from ai_content_detector.detectors import text_detectors, style_detector
    from ai_content_detector.rl_evasion.text_evasion import rewards
    from ml_pipeline.utils import utils

    class Ready:
        def __init__(self, device="auto", model_name=None):
            assert device == "cpu"

        def is_available(self):
            return True

        def detect(self, text):
            assert len(text.split()) > 100
            return SimpleNamespace(score=0.3)

    class Invalid(Ready):
        def detect(self, text):
            return SimpleNamespace(score=float("nan"))

    class NoClassifier(Ready):
        def is_available(self):
            return False

    class NoSupport(Ready):
        def detect(self, text):
            raise RuntimeError("Support centroids are missing")

    monkeypatch.setattr(text_detectors, "BinocularsDetector", Ready)
    monkeypatch.setattr(text_detectors, "FastDetectGPTDetector", Invalid)
    monkeypatch.setattr(text_detectors, "DivEyeDetector", NoClassifier)
    monkeypatch.setattr(text_detectors, "ModernBERTDetector", Ready)
    monkeypatch.setattr(text_detectors, "TFIDFDetector", lambda model_name: Ready(device="cpu"))
    monkeypatch.setattr(style_detector, "StyleEmbeddingDetector", NoSupport)
    monkeypatch.setattr(rewards, "_build_roberta_detector_reward", lambda device: lambda text: 0.8)
    monkeypatch.setattr(utils, "_load_index", lambda path: {
        "one": {"model_name": "qlora_one", "artifact_type": "hf_dir", "checkpoint_id": "one"},
        "two": {"model_name": "qlora_two", "artifact_type": "hf_dir", "checkpoint_id": "two"},
        "three": {"model_name": "tfidf_logreg", "artifact_type": "joblib", "checkpoint_id": "three"},
    })
    zoo = meta_adapt.DetectorZoo.build_default(device="cpu")
    assert set(zoo.detectors) == {"binoculars", "roberta_openai", "qlora_one", "qlora_two", "tfidf_logreg"}
    assert zoo.detectors["roberta_openai"]("text") == pytest.approx(0.2)
    assert zoo.availability["qlora_one"]["checkpoint_id"] == "one"
    assert zoo.availability["diveye"]["status"] == "unavailable"
    assert "Support centroids" in zoo.availability["style_embedding"]["reason"]
    assert "invalid AI score" in zoo.availability["fast_detect_gpt"]["reason"]


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = torch.nn.Parameter(torch.tensor(0.9))

    def save_pretrained(self, path):
        pass


def make_adaptation(path):
    zoo = meta_adapt.DetectorZoo(seed=17)
    zoo.add("a", lambda text: 0.2)
    zoo.add("b", lambda text: 0.8)
    adaptation = meta_adapt.MAMLAdaptation(
        TinyPolicy(), SimpleNamespace(save_pretrained=lambda path: None), zoo,
        meta_lr=0.02, inner_lr=0.01, outer_steps=3, inner_steps=2,
        detectors_per_episode=2, output_dir=str(path), seed=17,
    )

    def loss(self, model, detector, prompts, fast_weights=None, num_samples=1):
        value = model.policy if fast_weights is None else fast_weights["policy"]
        coefficient = torch.rand(()) + float(np.random.random()) + 0.5
        return coefficient * (value - detector("sample")).square()

    def sample(self, model, prompt, fast_weights, seed):
        return "sample", model.policy * 0

    adaptation._compute_evasion_loss = MethodType(loss, adaptation)
    adaptation._sample_completion = MethodType(sample, adaptation)
    return adaptation


def test_second_order_meta_restart_preserves_partial_gradients_and_optimizer(tmp_path, monkeypatch):
    seed_everything(17)
    baseline = make_adaptation(tmp_path / "baseline")
    baseline.train(["p0", "p1", "p2", "p3"])
    expected = load_checkpoint(tmp_path / "baseline" / "meta_restart.pt")

    seed_everything(17)
    interrupted = make_adaptation(tmp_path / "resumed")
    save = meta_adapt.atomic_save

    def interrupt(state, path):
        save(state, path)
        episode = state.get("episode")
        if state.get("outer_step") == 1 and episode and episode["next_detector"] == 1:
            raise CheckpointPause("simulated scheduler interruption")

    monkeypatch.setattr(meta_adapt, "atomic_save", interrupt)
    with pytest.raises(CheckpointPause):
        interrupted.train(["p0", "p1", "p2", "p3"])
    partial = load_checkpoint(tmp_path / "resumed" / "meta_restart.pt")
    assert partial["gradients"]["policy"].abs() > 0
    monkeypatch.setattr(meta_adapt, "atomic_save", save)
    resumed = make_adaptation(tmp_path / "resumed")
    resumed.train(["p0", "p1", "p2", "p3"])
    actual = load_checkpoint(tmp_path / "resumed" / "meta_restart.pt")
    assert torch.equal(actual["model"]["policy"], expected["model"]["policy"])
    assert actual["losses"] == expected["losses"]
    assert resumed.training_history == baseline.training_history
    assert torch.equal(actual["initial"]["policy"], expected["initial"]["policy"])
    for key in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(actual["optimizer"]["state"][0][key],
                           expected["optimizer"]["state"][0][key])


def test_meta_restart_rejects_changed_prompt_manifest(tmp_path):
    adaptation = make_adaptation(tmp_path)
    adaptation.train(["p0", "p1", "p2", "p3"])
    with pytest.raises(ValueError, match="configuration differs"):
        make_adaptation(tmp_path).train(["changed", "p1", "p2", "p3"])


def test_adaptation_restart_and_disjoint_evaluation(tmp_path, monkeypatch):
    seed_everything(17)
    baseline = make_adaptation(tmp_path / "baseline")
    expected = baseline.measure_adaptation_cost(
        lambda text: 0.4, "heldout", ["adapt"], max_steps=6,
        measure_interval=2, evaluation_prompts=["eval"],
    )
    seed_everything(17)
    interrupted = make_adaptation(tmp_path / "resumed")
    initial = interrupted.model.policy.detach().clone()
    save = meta_adapt.atomic_save

    def interrupt(state, path):
        save(state, path)
        if state.get("step") == 3:
            raise CheckpointPause("simulated interruption")

    monkeypatch.setattr(meta_adapt, "atomic_save", interrupt)
    with pytest.raises(CheckpointPause):
        interrupted.measure_adaptation_cost(
            lambda text: 0.4, "heldout", ["adapt"], max_steps=6,
            measure_interval=2, evaluation_prompts=["eval"],
        )
    assert torch.equal(interrupted.model.policy.detach(), initial)
    monkeypatch.setattr(meta_adapt, "atomic_save", save)
    resumed = make_adaptation(tmp_path / "resumed")
    actual = resumed.measure_adaptation_cost(
        lambda text: 0.4, "heldout", ["adapt"], max_steps=6,
        measure_interval=2, evaluation_prompts=["eval"],
    )
    assert actual == expected
    assert actual["steps"][-1] == 6
    a = load_checkpoint(tmp_path / "resumed" / "adapt_heldout.pt")
    b = load_checkpoint(tmp_path / "baseline" / "adapt_heldout.pt")
    assert torch.equal(a["model"]["policy"], b["model"]["policy"])
    with pytest.raises(ValueError, match="must be disjoint"):
        resumed.measure_adaptation_cost(
            lambda text: 0.4, "overlap", ["same"], evaluation_prompts=["same"],
        )


def test_adaptation_speedup_is_undefined_when_threshold_not_reached(tmp_path):
    adaptation = make_adaptation(tmp_path)
    assert adaptation._compute_speedup(
        {"steps": [0, 5], "evasion": [0.2, 0.6]},
        {"steps": [0, 5], "evasion": [0.2, 0.5]},
    ) is None
    assert adaptation._compute_speedup(
        {"steps": [0, 5], "evasion": [0.8, 0.9]},
        {"steps": [0, 5], "evasion": [0.8, 0.9]},
    ) is None
    assert adaptation._compute_speedup(
        {"steps": [0, 5, 10], "evasion": [0.2, 0.8, 0.9]},
        {"steps": [0, 5, 10], "evasion": [0.2, 0.3, 0.8]},
    ) == 2.0


def test_branch_accumulation_preserves_closed_form_second_order_gradient(tmp_path):
    adaptation = make_adaptation(tmp_path)
    adaptation.outer_steps = 1

    def loss(self, model, detector, prompts, fast_weights=None, num_samples=1):
        value = model.policy if fast_weights is None else fast_weights["policy"]
        return (value - detector("sample")).square()

    adaptation._compute_evasion_loss = MethodType(loss, adaptation)
    adaptation.train(["p0", "p1", "p2", "p3"])
    state = load_checkpoint(tmp_path / "meta_restart.pt")
    contraction = (1 - 2 * adaptation.inner_lr) ** adaptation.inner_steps
    expected_gradient = (2 * 0.9 - 0.2 - 0.8) * contraction ** 2
    measured_gradient = float(state["optimizer"]["state"][0]["exp_avg"]) / 0.1
    assert measured_gradient == pytest.approx(expected_gradient, abs=1e-7)


@pytest.mark.parametrize("architecture", ["gpt2", "qwen35"])
def test_native_causal_model_supports_full_second_order_meta_step(tmp_path, architecture):
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers.tokenization_utils_base import BatchEncoding

    class Tokenizer:
        eos_token_id = 19

        def __call__(self, *args, **kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[1, 2]])})

        def decode(self, tokens, **kwargs):
            return " ".join(str(token) for token in tokens.tolist())

        def save_pretrained(self, path):
            pass

    seed_everything(17)
    if architecture == "gpt2":
        config = GPT2Config(vocab_size=20, n_embd=8, n_layer=1, n_head=2,
                            bos_token_id=1, eos_token_id=19, pad_token_id=19)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config)
    else:
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        config = Qwen3_5TextConfig(
            vocab_size=20, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, layer_types=["linear_attention", "full_attention"],
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=1, linear_num_value_heads=1,
            bos_token_id=1, eos_token_id=19, pad_token_id=19,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
        )
        config._attn_implementation = "eager"
        model = Qwen3_5ForCausalLM(config)
    zoo = meta_adapt.DetectorZoo(seed=17)
    zoo.add("a", lambda text: sum(map(int, text.split())) % 10 / 10)
    zoo.add("b", lambda text: 1 - sum(map(int, text.split())) % 10 / 10)
    adaptation = meta_adapt.MAMLAdaptation(
        model, Tokenizer(), zoo, outer_steps=1, inner_steps=1,
        max_new_tokens=2, output_dir=str(tmp_path), seed=17,
    )
    adaptation.train(["p0", "p1", "p2", "p3"])
    state = load_checkpoint(tmp_path / "meta_restart.pt")
    assert state["outer_step"] == 1
    assert np.isfinite(state["losses"][0])
    assert any(item["exp_avg"].abs().sum() > 0 for item in state["optimizer"]["state"].values())
