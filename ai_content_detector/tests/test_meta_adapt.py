from __future__ import annotations

import types

import pytest
import torch

from ai_content_detector.rl_evasion.arms_race.meta_adapt import DetectorZoo, MAMLAdaptation


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = torch.nn.Parameter(torch.tensor(0.0))


def _policy_loss(detector):
    adaptation = object.__new__(MAMLAdaptation)
    adaptation.rollouts_per_prompt = 2
    adaptation.seed = 7
    calls = iter([("good", 1.0), ("bad", -1.0)])

    def sample(self, model, prompt, fast_weights, seed):
        text, sign = next(calls)
        return text, sign * model.policy

    adaptation._sample_completion = types.MethodType(sample, adaptation)
    model = TinyModel()
    loss = adaptation._compute_evasion_loss(model, detector, ["prompt"], num_samples=1)
    loss.backward()
    return float(model.policy.grad)


def test_policy_gradient_favors_lower_detector_score():
    grad = _policy_loss(lambda text: 0.0 if text == "good" else 1.0)
    assert grad < 0.0


def test_reversing_detector_scores_reverses_gradient():
    grad = _policy_loss(lambda text: 1.0 if text == "good" else 0.0)
    assert grad > 0.0


def test_invalid_detector_score_fails_closed():
    with pytest.raises(ValueError, match="invalid AI score"):
        _policy_loss(lambda text: 2.0)


def test_empty_detector_zoo_cannot_be_sampled():
    with pytest.raises(ValueError, match="empty detector zoo"):
        DetectorZoo().sample()
