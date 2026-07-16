from __future__ import annotations

import pytest
import torch

from ai_content_detector.detectors.text_detectors import FastDetectGPTDetector


def test_standardized_curvature_matches_manual_calculation():
    scoring_logits = torch.log(torch.tensor([[[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]]))
    reference_logits = torch.log(torch.tensor([[[0.5, 0.4, 0.1], [0.1, 0.7, 0.2]]]))
    targets = torch.tensor([[0, 1]])
    mask = torch.ones((1, 2))

    result = FastDetectGPTDetector._criterion_from_logits(
        scoring_logits, reference_logits, targets, mask,
    )

    logp = torch.log_softmax(scoring_logits, dim=-1)
    refp = torch.softmax(reference_logits, dim=-1)
    observed = torch.tensor([logp[0, 0, 0], logp[0, 1, 1]])
    expected = (refp * logp).sum(dim=-1).squeeze(0)
    variance = (refp * logp.square()).sum(dim=-1).squeeze(0) - expected.square()
    manual = (observed - expected).sum() / variance.sum().sqrt()
    assert result.item() == pytest.approx(manual.item())


def test_padding_is_excluded_from_numerator_and_variance():
    logits = torch.log(torch.tensor([[[0.8, 0.2], [0.01, 0.99]]]))
    targets = torch.tensor([[0, 0]])
    masked = FastDetectGPTDetector._criterion_from_logits(
        logits, logits, targets, torch.tensor([[1.0, 0.0]]),
    )
    first_only = FastDetectGPTDetector._criterion_from_logits(
        logits[:, :1], logits[:, :1], targets[:, :1], torch.ones((1, 1)),
    )
    assert masked.item() == pytest.approx(first_only.item())
