"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ai_content_detector.rl_evasion.text_evasion.evaluate import (
    compute_auroc,
    compute_tpr_at_fpr,
    evaluate_text_evasion,
)


# ---------------------------------------------------------------------------
# evaluate_text_evasion
# ---------------------------------------------------------------------------


class TestEvaluateTextEvasion:
    @staticmethod
    def _make_reward_fn(evasion: float = 0.6, semantic: float = 0.85):
        """Return a mock reward_fn that returns fixed scores."""

        class MockReward:
            def __call__(self, source, generated):
                # Quality: always 1.0 for simplicity
                total = 0.6 * evasion + 0.3 * semantic + 0.1 * 1.0
                return {
                    "total": total,
                    "evasion": evasion,
                    "semantic": semantic,
                    "quality": 1.0,
                    "per_detector": {"det_a": evasion},
                }

        return MockReward()

    def test_basic_metrics(self):
        reward_fn = self._make_reward_fn(evasion=0.7, semantic=0.9)
        metrics = evaluate_text_evasion(
            generated_texts=["gen1", "gen2", "gen3"],
            source_texts=["src1", "src2", "src3"],
            reward_fn=reward_fn,
        )
        assert metrics["mean_evasion"] == pytest.approx(0.7)
        assert metrics["mean_semantic_similarity"] == pytest.approx(0.9)
        assert metrics["num_samples"] == 3

    def test_attack_success_rate_all_evaded(self):
        reward_fn = self._make_reward_fn(evasion=0.8)
        metrics = evaluate_text_evasion(["g1", "g2"], ["s1", "s2"], reward_fn)
        # evasion > 0.5 for all -> 100%
        assert metrics["attack_success_rate"] == pytest.approx(1.0)

    def test_attack_success_rate_none_evaded(self):
        reward_fn = self._make_reward_fn(evasion=0.3)
        metrics = evaluate_text_evasion(["g1", "g2"], ["s1", "s2"], reward_fn)
        assert metrics["attack_success_rate"] == pytest.approx(0.0)

    def test_per_detector_metrics(self):
        reward_fn = self._make_reward_fn(evasion=0.6)
        metrics = evaluate_text_evasion(["g1"], ["s1"], reward_fn)
        assert "evasion_det_a" in metrics
        assert metrics["evasion_det_a"] == pytest.approx(0.6)

    def test_mean_detector_ai_score(self):
        reward_fn = self._make_reward_fn(evasion=0.7)
        metrics = evaluate_text_evasion(["g1"], ["s1"], reward_fn)
        # detector score = 1 - evasion = 0.3
        assert metrics["mean_detector_ai_score"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# compute_tpr_at_fpr
# ---------------------------------------------------------------------------


class TestComputeTPRAtFPR:
    def test_perfect_detector(self):
        # Perfect separation: AI=1.0, Human=0.0
        scores = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        labels = [1, 1, 1, 0, 0, 0]
        tpr = compute_tpr_at_fpr(scores, labels, fpr_target=0.01)
        assert tpr == pytest.approx(1.0)

    def test_random_detector(self):
        rng = np.random.RandomState(42)
        n = 1000
        scores = rng.rand(n).tolist()
        labels = rng.randint(0, 2, n).tolist()
        tpr = compute_tpr_at_fpr(scores, labels, fpr_target=0.5)
        # Random detector: TPR ~ FPR
        assert 0.2 < tpr < 0.8

    def test_fpr_target_zero(self):
        scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
        labels = [1, 1, 1, 0, 0, 0]
        tpr = compute_tpr_at_fpr(scores, labels, fpr_target=0.0)
        assert 0.0 <= tpr <= 1.0


# ---------------------------------------------------------------------------
# compute_auroc
# ---------------------------------------------------------------------------


class TestComputeAUROC:
    def test_perfect_separation(self):
        scores = [1.0, 1.0, 0.0, 0.0]
        labels = [1, 1, 0, 0]
        assert compute_auroc(scores, labels) == pytest.approx(1.0)

    def test_inverse_separation(self):
        scores = [0.0, 0.0, 1.0, 1.0]
        labels = [1, 1, 0, 0]
        assert compute_auroc(scores, labels) == pytest.approx(0.0)

    def test_random_near_half(self):
        rng = np.random.RandomState(42)
        n = 1000
        scores = rng.rand(n).tolist()
        labels = rng.randint(0, 2, n).tolist()
        auroc = compute_auroc(scores, labels)
        assert 0.4 < auroc < 0.6

    def test_single_class(self):
        # sklearn returns nan (with warning) for single-class y_true;
        # compute_auroc catches ValueError but not the nan/warning path.
        # Verify it doesn't crash.
        scores = [0.5, 0.6, 0.7]
        labels = [1, 1, 1]
        result = compute_auroc(scores, labels)
        assert isinstance(result, float)

    def test_two_samples(self):
        assert compute_auroc([1.0, 0.0], [1, 0]) == pytest.approx(1.0)
