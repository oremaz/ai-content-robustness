import random
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ai_content_detector.rl_evasion.benchmarking import benchmark
from ai_content_detector.rl_evasion.benchmarking.baselines import BaseEvasionMethod, EvasionCapability
from ai_content_detector.rl_evasion.benchmarking.baselines import (
    PromptEngineeringBaseline, SamplingPerturbationBaseline,
)
from ai_content_detector.rl_evasion.benchmarking.datasets import BenchmarkDataset
from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, load_checkpoint
from ai_content_detector.rl_evasion.config import seed_everything


class StochasticRewrite(BaseEvasionMethod):
    name = "stochastic"
    capabilities = frozenset({EvasionCapability.REWRITE})

    def __init__(self):
        self._seed = 17
        self._rng = random.Random(self._seed)
        self.calls = []

    def evade(self, text):
        self.calls.append(text)
        return f"{text} {self._rng.random()} {np.random.random()} {float(torch.rand(()))}"


def make_runner(path):
    method = StochasticRewrite()
    dataset = BenchmarkDataset(name="fixture", prompts=[f"p{i}" for i in range(5)],
                               ai_texts=[f"a{i}" for i in range(5)])
    runner = benchmark.BenchmarkRunner([method], dataset, checkpoint_dir=str(path))
    score_calls = []

    def score(source, generated):
        score_calls.append(generated)
        value = float(torch.rand(()))
        return {"total": value, "evasion": value, "semantic": 0.8,
                "per_detector": {"fixture": value}}

    runner.reward_fn = score
    return runner, method, score_calls


def test_required_detector_pool_cannot_silently_shrink(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path)
    runner.detector_names = ["working", "broken"]
    runner.require_all_detectors = True

    def factory(name, device):
        if name == "broken":
            raise RuntimeError("missing checkpoint")
        return SimpleNamespace(name=name)

    monkeypatch.setattr(benchmark, "build_detector_reward_from_name", factory)
    with pytest.raises(RuntimeError, match="Required eval detector broken"):
        runner.setup_detectors()
    runner.require_all_detectors = False
    runner.setup_detectors()
    runner.save_results(str(tmp_path))
    saved = json.loads((tmp_path / "benchmark_results.json").read_text())
    assert saved["detectors"] == ["working", "broken"]
    assert saved["effective_detectors"] == ["working"]
    assert saved["effective_heldout_detectors"] == []


@pytest.mark.parametrize("phase", ["generation", "scoring"])
def test_benchmark_resume_preserves_generated_outputs_and_cached_scores(tmp_path, monkeypatch, phase):
    seed_everything(17)
    baseline, _, _ = make_runner(tmp_path / "baseline")
    expected = baseline.run()["stochastic"]
    seed_everything(17)
    interrupted, _, _ = make_runner(tmp_path / "resumed")
    save = benchmark.atomic_save

    def interrupt(state, path):
        save(state, path)
        application = state["applications"].get("stochastic", {})
        if ((phase == "generation" and application.get("attempted") == 3)
                or (phase == "scoring" and len(state["scores"]) == 2)):
            raise CheckpointPause("simulated interruption")

    monkeypatch.setattr(benchmark, "atomic_save", interrupt)
    with pytest.raises(CheckpointPause):
        interrupted.run()
    monkeypatch.setattr(benchmark, "atomic_save", save)
    resumed, method, scoring = make_runner(tmp_path / "resumed")
    actual = resumed.run()["stochastic"]
    assert {k: v for k, v in actual.items() if k != "wall_time_sec"} == {
        k: v for k, v in expected.items() if k != "wall_time_sec"
    }
    assert len(method.calls) == (2 if phase == "generation" else 0)
    assert len(scoring) == (5 if phase == "generation" else 3)
    a = load_checkpoint(tmp_path / "resumed" / "benchmark_restart.pt")
    b = load_checkpoint(tmp_path / "baseline" / "benchmark_restart.pt")
    assert a["applications"] == b["applications"]
    assert a["scores"] == b["scores"]


def test_completed_benchmark_skips_transforms_and_scoring(tmp_path):
    runner, _, _ = make_runner(tmp_path)
    runner.run()
    resumed, method, score_calls = make_runner(tmp_path)
    resumed.run()
    assert not method.calls
    assert not score_calls


def test_benchmark_restart_rejects_changed_sample_content(tmp_path):
    runner, _, _ = make_runner(tmp_path)
    runner.run()
    resumed, _, _ = make_runner(tmp_path)
    resumed.dataset.ai_texts[0] = "different content"
    with pytest.raises(ValueError, match="configuration differs"):
        resumed.run()


@pytest.mark.parametrize("method_class", [PromptEngineeringBaseline, SamplingPerturbationBaseline])
def test_native_hf_sampling_is_seeded_without_changing_caller_rng(method_class):
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers.tokenization_utils_base import BatchEncoding

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[1, 2]]),
                                  "attention_mask": torch.ones((1, 2), dtype=torch.long)})

        def decode(self, tokens, **kwargs):
            return str(tokens.tolist())

    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=20, n_embd=16, n_layer=1, n_head=2,
        bos_token_id=1, eos_token_id=19, pad_token_id=19,
    )).eval()
    method = method_class(device="cpu", max_new_tokens=4, seed=17)
    method._model, method._tokenizer = model, Tokenizer()
    before = torch.get_rng_state().clone()
    first = method.generate("prompt")
    assert torch.equal(torch.get_rng_state(), before)
    torch.rand(100)
    assert method.generate("prompt") == first


def test_benchmark_releases_generator_before_loading_next_method(tmp_path):
    methods = []

    class LoadedMethod(StochasticRewrite):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self._model = None

        def _load(self):
            assert all(other._model is None for other in methods if other is not self)
            self._model = object()

    methods.extend([LoadedMethod("first"), LoadedMethod("second")])
    dataset = BenchmarkDataset(name="fixture", prompts=["p"], ai_texts=["a"])
    runner = benchmark.BenchmarkRunner(methods, dataset, checkpoint_dir=str(tmp_path))
    runner.reward_fn = lambda source, generated: {"total": 0.6, "evasion": 0.7, "semantic": 0.8}
    assert set(runner.run()) == {"first", "second"}
    assert all(method._model is None for method in methods)
