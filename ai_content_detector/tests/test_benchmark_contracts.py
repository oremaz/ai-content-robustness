from __future__ import annotations

import pytest

from ai_content_detector.rl_evasion.benchmarking.baselines import (
    BaseEvasionMethod,
    EvasionCapability,
)
from ai_content_detector.rl_evasion.benchmarking.benchmark import BenchmarkRunner
from ai_content_detector.rl_evasion.benchmarking.datasets import BenchmarkDataset


class RewriteMethod(BaseEvasionMethod):
    name = "rewrite"
    capabilities = frozenset({EvasionCapability.REWRITE})

    def evade(self, text: str) -> str:
        if text == "bad":
            raise RuntimeError("sample failed")
        return f"rewritten:{text}"


class GenerateMethod(BaseEvasionMethod):
    name = "generate"
    capabilities = frozenset({EvasionCapability.GENERATE})

    def generate(self, prompt: str) -> str:
        return f"generated:{prompt}"


def test_generation_method_is_skipped_on_rewrite_dataset():
    dataset = BenchmarkDataset(
        name="rewrite",
        prompts=["prompt"],
        ai_texts=["ai text"],
        human_references=["human text"],
    )
    application = BenchmarkRunner([GenerateMethod()], dataset)._apply_method(GenerateMethod())
    assert application.status == "skipped"
    assert application.generated_texts == []


def test_rewrite_failures_are_not_scored_as_fallback_outputs():
    dataset = BenchmarkDataset(
        name="rewrite",
        prompts=["p1", "p2"],
        ai_texts=["good", "bad"],
        human_references=["h1", "h2"],
    )
    application = BenchmarkRunner([RewriteMethod()], dataset)._apply_method(RewriteMethod())
    assert application.status == "partial"
    assert application.generated_texts == ["rewritten:good"]
    assert application.source_texts == ["good"]
    assert application.sample_indices == [0]
    assert application.errors[0]["sample_index"] == 1


def test_generation_uses_human_reference_for_quality():
    dataset = BenchmarkDataset(
        name="generation",
        prompts=["prompt"],
        ai_texts=[""],
        human_references=["reference continuation"],
        ai_texts_available=False,
    )
    application = BenchmarkRunner([GenerateMethod()], dataset)._apply_method(GenerateMethod())
    assert application.generated_texts == ["generated:prompt"]
    assert application.source_texts == ["reference continuation"]


def test_dataset_rejects_misaligned_fields():
    with pytest.raises(ValueError, match="equal lengths"):
        BenchmarkDataset(name="bad", prompts=["p"], ai_texts=[])


def test_detector_setup_fails_when_every_detector_fails(monkeypatch):
    dataset = BenchmarkDataset(name="rewrite", prompts=["p"], ai_texts=["a"])
    runner = BenchmarkRunner([RewriteMethod()], dataset, detector_names=["missing"])

    def fail(*args, **kwargs):
        raise RuntimeError("missing")

    monkeypatch.setattr(
        "ai_content_detector.rl_evasion.benchmarking.benchmark.build_detector_reward_from_name",
        fail,
    )
    with pytest.raises(RuntimeError, match="No eval detector"):
        runner.setup_detectors()
