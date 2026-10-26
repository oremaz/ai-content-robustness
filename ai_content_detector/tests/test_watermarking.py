"""Numerical and threat-model checks for watermark distillation experiments."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ai_content_detector.watermarking.protocol import (
    STAGES, detection_metrics, operating_point, validate_partitions,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "results/research_workflow/watermarking/resources"


@pytest.fixture(scope="module")
def official():
    if not (SOURCES / "textseal").exists():
        pytest.skip("Official watermark source has not been prepared")
    sys.path.insert(0, str(SOURCES / "textseal"))
    from textseal.watermarking.core import prf_binary
    return prf_binary


def test_forward_kl_normalization_padding_and_gradient(official):
    from ai_content_detector.watermarking.text import distillation_loss
    config = SimpleNamespace(ngram=1, secret_key=1234, gamma=.25, delta=2.)
    teacher = torch.randn(2, 3, 17, requires_grad=True)
    student = torch.randn(2, 3, 17, requires_grad=True)
    contexts = torch.tensor([[[2], [3], [4]], [[5], [6], [7]]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    green = official(contexts.reshape(-1, 1), torch.arange(17).expand(6, -1), config.secret_key, config.gamma)
    target = (teacher.detach().reshape(6, 17) + 2*green).softmax(-1)
    expected = F.kl_div(student.reshape(6, 17)[mask.flatten()].log_softmax(-1),
                        target[mask.flatten()], reduction="sum") / 3
    observed = distillation_loss(student, teacher, contexts, mask, config, chunk=2)
    torch.testing.assert_close(observed, expected)
    observed.backward()
    assert teacher.grad is None
    assert student.grad[~mask].abs().sum() == 0
    torch.testing.assert_close(student.grad[mask], (student.detach().softmax(-1)[mask]-target.reshape(2,3,17)[mask])/3)


def test_distillation_target_has_no_signal_without_bias(official):
    from ai_content_detector.watermarking.text import distillation_loss
    logits = torch.randn(1, 5, 17)
    config = SimpleNamespace(ngram=1, secret_key=1234, gamma=.25, delta=0.)
    loss = distillation_loss(logits, logits, torch.ones(1,5,1,dtype=torch.long),
                             torch.ones(1,5,dtype=torch.bool), config)
    assert abs(loss.item()) < 1e-6


def test_detection_counts_use_strict_frozen_threshold():
    threshold = operating_point([.1, .2, .2, .3], fpr=.25)
    result = detection_metrics([.3, .9], [.1, .2], threshold)
    assert result["detection"]["count"] == 1
    assert result["false_positive"]["count"] == 0
    assert result["detection"]["ci95"][0] < .5 < result["detection"]["ci95"][1]


def test_partitions_reject_question_and_image_leakage():
    data = {"text": {"train": [{"id":"a"}], "test": [{"id":"b"}]},
            "images": {"train": ["a"], "validation": ["b"], "test": ["c"]}}
    validate_partitions(data)
    data["text"]["test"][0]["id"] = "a"
    with pytest.raises(ValueError, match="Text document leakage"):
        validate_partitions(data)
    data["text"]["test"][0]["id"] = "b"
    data["images"]["test"] = ["a"]
    with pytest.raises(ValueError, match="Image leakage"):
        validate_partitions(data)


def test_removal_jobs_wait_for_their_watermarked_control():
    assert STAGES["text_clean"] == ["distilled"]
    assert STAGES["text_base"] == ["distilled"]
    for family in ("rar", "dcae"):
        assert STAGES[f"{family}_clean"] == [family]
        assert STAGES[f"{family}_base"] == [family]


def test_image_domain_metrics_use_matching_negative_classes():
    from ai_content_detector.watermarking.image import summarize
    rows = [{"label": 0, "bit_accuracy": .8, "clip_alignment": .3},
            {"label": 1, "bit_accuracy": .8, "clip_alignment": .3}]
    negative = [{"label": 0, "bit_accuracy": .9, "clip_alignment": .3},
                {"label": 1, "bit_accuracy": .2, "clip_alignment": .3}]
    result = summarize(rows, negative, .75)
    assert result["all"]["false_positive"]["rate"] == .5
    assert result["class_zero"]["false_positive"]["rate"] == 1.
    assert result["other_classes"]["false_positive"]["rate"] == 0.


def test_qwen_hybrid_cache_matches_full_prefix(monkeypatch):
    import inspect
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from ai_content_detector.watermarking.text import configured_cache
    for name in ("causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        monkeypatch.setattr(qwen, name, inspect.unwrap(getattr(qwen, name)))
    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"], eos_token_id=63, pad_token_id=62)
    model = qwen.Qwen3_5ForCausalLM(config).eval()
    with torch.no_grad():
        first = model(torch.tensor([[1,2,3,4]]), past_key_values=configured_cache(model), use_cache=True)
        incremental = model(torch.tensor([[5]]), past_key_values=first.past_key_values, use_cache=True).logits
        full = model(torch.tensor([[1,2,3,4,5]]), use_cache=False).logits[:, -1:]
    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-5)
