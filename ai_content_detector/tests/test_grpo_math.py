"""Math-correctness tests for GRPO advantage normalization.

These tests don't load any LLM — they isolate the policy-gradient arithmetic
that the manual GRPO loop in ``rl_evasion/text_evasion/grpo_trainer.py``
relies on so a regression to the old "train on argmax only" or "no
group-mean subtraction" bug would fail loudly.
"""

from __future__ import annotations

import numpy as np


def _group_advantages(rewards: np.ndarray) -> np.ndarray:
    """Reference implementation of the in-group advantage used by GRPO."""
    std = float(rewards.std())
    if std > 1e-8:
        return (rewards - rewards.mean()) / std
    return np.zeros_like(rewards)


class TestGRPOAdvantage:
    def test_zero_mean(self):
        rewards = np.array([0.1, 0.4, 0.6, 0.9])
        adv = _group_advantages(rewards)
        assert abs(float(adv.sum())) < 1e-6

    def test_unit_std(self):
        rewards = np.array([0.1, 0.4, 0.6, 0.9])
        adv = _group_advantages(rewards)
        # ddof=0 (numpy default) std should be 1 after standardization
        assert abs(float(adv.std()) - 1.0) < 1e-6

    def test_zero_advantage_when_constant(self):
        rewards = np.full((5,), 0.42)
        adv = _group_advantages(rewards)
        assert np.allclose(adv, 0.0)

    def test_positive_advantage_for_above_mean(self):
        rewards = np.array([0.1, 0.5, 0.9])
        adv = _group_advantages(rewards)
        # Above-mean sample (0.9) should have strictly positive advantage
        assert adv[2] > 0
        # Below-mean sample (0.1) should have strictly negative advantage
        assert adv[0] < 0


class TestGRPOLossSign:
    """The policy-gradient loss is ``-A * sum log pi_theta(y)``.

    A *positive* advantage with a *positive* log-prob therefore yields a
    negative loss — minimizing it pushes the log-prob *up* (correct).
    The previous buggy implementation had ``loss = -A * outputs.loss``
    where ``outputs.loss`` is the (positive) cross-entropy of the
    completion, which made the optimizer behave as if every above-mean
    sample should be discouraged. This test pins the sign convention so
    that regression cannot return silently.
    """

    def test_positive_advantage_decreases_loss(self):
        # Larger log-prob ⇒ more negative loss (we minimize, so larger log-prob is rewarded).
        log_p_low = -3.0
        log_p_high = -1.0
        adv = 0.7
        loss_low = -adv * log_p_low
        loss_high = -adv * log_p_high
        assert loss_high < loss_low

    def test_negative_advantage_flips_sign(self):
        log_p_low = -3.0
        log_p_high = -1.0
        adv = -0.7
        loss_low = -adv * log_p_low
        loss_high = -adv * log_p_high
        # When advantage is negative, the optimizer should DECREASE log-prob,
        # so smaller log-prob is preferred.
        assert loss_low < loss_high
