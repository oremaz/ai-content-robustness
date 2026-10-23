from __future__ import annotations

import pytest
import json
import hashlib

from ai_content_detector.rl_evasion.benchmarking.baselines import (
    BaseEvasionMethod,
    EvasionCapability,
)
from ai_content_detector.rl_evasion.benchmarking.benchmark import BenchmarkRunner
from ai_content_detector.rl_evasion.benchmarking.datasets import BenchmarkDataset, load_hc3


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


def test_local_benchmark_pool_excludes_api_even_when_configured(monkeypatch):
    from ai_content_detector.rl_evasion.benchmarking.baselines import (
        get_all_baselines, get_lightweight_baselines, OpenRouterBaseline,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    for methods in (get_all_baselines(include_api=False), get_lightweight_baselines(include_api=False)):
        assert not any(isinstance(method, OpenRouterBaseline) for method in methods)


def test_hc3_json_loader_preserves_order_answers_and_source_revision(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    revision = "a" * 40
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    data = [
        {"question": "skipped", "human_answers": ["human"], "chatgpt_answers": []},
        {"question": "first", "human_answers": ["h1", "h2"], "chatgpt_answers": ["a1", "a2"]},
        {"question": "second", "human_answers": ["h3"], "chatgpt_answers": ["a3"]},
        {"question": "third", "human_answers": ["h4"], "chatgpt_answers": ["a4"]},
    ]
    source = snapshot / "all.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in data))
    link = tmp_path / "hc3-all.jsonl"
    link.symlink_to(source)
    dataset = load_hc3(max_samples=2, data_file=str(link))
    assert dataset.prompts == ["first", "second"]
    assert dataset.ai_texts == ["a1", "a3"]
    assert dataset.human_references == ["h1", "h3"]
    assert dataset.sample_ids == ["HC3:train:1", "HC3:train:2"]
    assert dataset.source_revision == f"{revision}:sha256:{hashlib.sha256(source.read_bytes()).hexdigest()}"


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
