"""Probability-ratio tilting on the actually exposed API support."""

import pytest
import torch

from ai_content_detector.watermarking.text_attacks import TopKResponse, ratio_api_choice


def test_ratio_tilt_uses_surrogate_probabilities_and_blocks_source_completions(monkeypatch):
    target = TopKResponse(torch.tensor([[7, 3, 9]]), torch.tensor([[.6, .3, .1]]), 10)
    reference = torch.tensor([[.2, .3, .5]])
    observed = []

    def choose(probabilities, count, generator=None):
        observed.append(probabilities)
        return probabilities.argmax(-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", choose)
    choice, fallback = ratio_api_choice(target, reference, {3}, exponent=2.)
    expected = torch.tensor([[.2**2/.6, 0., .5**2/.1]], dtype=torch.float64)
    assert torch.allclose(observed[0], expected/expected.sum(), atol=1.e-7)
    assert choice.tolist() == [9] and fallback.tolist() == [False]
    ratio_api_choice(target, reference, exponent=1.)
    assert torch.allclose(observed[1], reference.double(), atol=1.e-7)


def test_ratio_tilt_records_an_empty_blocked_support_and_rejects_invalid_inputs():
    target = TopKResponse(torch.tensor([[7, 3]]), torch.tensor([[1., 0.]]), 10)
    choice, fallback = ratio_api_choice(target, torch.tensor([[.5, .5]]), {7})
    assert choice.tolist() == [7] and fallback.tolist() == [True]
    with pytest.raises(ValueError, match="common positive support"):
        ratio_api_choice(target, torch.tensor([[0., 1.]]))
    with pytest.raises(ValueError, match="finite ratio"):
        ratio_api_choice(target, torch.tensor([[.5, .5]]), exponent=float("inf"))
