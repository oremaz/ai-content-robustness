"""Tests for reward functions (DetectorReward, CompositeReward)."""

from __future__ import annotations

import pytest

from ai_content_detector.rl_evasion.text_evasion.rewards import (
    CompositeReward,
    DetectorReward,
)


def test_composite_reward_rejects_empty_detector_ensemble():
    with pytest.raises(ValueError, match="at least one detector"):
        CompositeReward(detector_rewards=[])


class TestDetectorReward:
    def test_basic_reward(self):
        dr = DetectorReward(lambda t: 0.8, name="test")
        assert dr("any text") == pytest.approx(0.2)

    def test_reward_is_one_minus_score(self):
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            dr = DetectorReward(lambda t, s=score: s, name="test")
            assert dr("text") == pytest.approx(1.0 - score)

    def test_failure_raises_by_default(self):
        """A failing detector must surface the error — silent fallback hides broken training runs."""

        def failing_fn(text):
            raise ValueError("boom")

        dr = DetectorReward(failing_fn, name="failing")
        with pytest.raises(ValueError):
            dr("text")

    def test_failure_silent_fallback_opt_in(self):
        """silent_fallback=True is an explicit opt-in for debugging only."""

        def failing_fn(text):
            raise ValueError("boom")

        dr = DetectorReward(failing_fn, name="failing", silent_fallback=True)
        assert dr("text") == pytest.approx(0.5)

    def test_name_preserved(self):
        dr = DetectorReward(lambda t: 0.5, name="my_detector")
        assert dr.name == "my_detector"


_LONG_GEN = (
    "This is a reasonably long generated output used in tests to avoid the "
    "min_output_tokens rejection path that penalizes degenerate short outputs."
)
_LONG_SRC = (
    "Source text of roughly similar length to the generated string above, "
    "so the length-ratio quality component stays near its maximum."
)


class TestCompositeReward:
    def _make_composite(
        self,
        detector_scores: list[float],
        semantic_sim: float = 0.9,
        weights: dict | None = None,
        min_output_tokens: int = 5,
    ) -> CompositeReward:
        detector_rewards = [
            DetectorReward(lambda t, s=s: s, name=f"det_{i}")
            for i, s in enumerate(detector_scores)
        ]

        class MockSemantic:
            def __call__(self, source, generated):
                return semantic_sim

        return CompositeReward(
            detector_rewards=detector_rewards,
            semantic_reward=MockSemantic(),
            weights=weights or {"evasion": 0.6, "semantic": 0.3, "quality": 0.1},
            min_output_tokens=min_output_tokens,
        )

    def test_composite_returns_dict(self):
        cr = self._make_composite([0.8])
        result = cr("source", "generated text here")
        assert "total" in result
        assert "evasion" in result
        assert "semantic" in result
        assert "quality" in result
        assert "per_detector" in result

    def test_evasion_score(self):
        # Detector score 0.8 -> evasion reward 0.2
        cr = self._make_composite([0.8])
        result = cr(_LONG_SRC, _LONG_GEN)
        assert result["evasion"] == pytest.approx(0.2)

    def test_multi_detector_evasion_averaged(self):
        # Detector scores [0.8, 0.4] -> evasion rewards [0.2, 0.6] -> mean 0.4
        cr = self._make_composite([0.8, 0.4])
        result = cr(_LONG_SRC, _LONG_GEN)
        assert result["evasion"] == pytest.approx(0.4)

    def test_semantic_passthrough(self):
        cr = self._make_composite([0.5], semantic_sim=0.85)
        result = cr(_LONG_SRC, _LONG_GEN)
        assert result["semantic"] == pytest.approx(0.85)

    def test_quality_same_length(self):
        # Same word count -> quality close to 1.0
        source = "one two three four five"
        generated = "six seven eight nine ten"
        cr = self._make_composite([0.5], min_output_tokens=1)
        result = cr(source, generated)
        assert result["quality"] == pytest.approx(1.0, abs=0.05)

    def test_quality_penalizes_length_mismatch(self):
        # Short-output path: min_output_tokens gate rejects outright to quality=0.
        source = "one two three four five"
        generated = "a"  # 1 word
        cr = self._make_composite([0.5], min_output_tokens=5)
        result = cr(source, generated)
        assert result["quality"] == pytest.approx(0.0)
        assert result["rejected"] is True

    def test_quality_quadratic_penalty_for_mismatch(self):
        # With the gate disabled, a 2× length ratio should yield quality=0 (quadratic).
        source = "one two three"
        generated = "four five six seven eight nine"  # ratio 2.0 → 1 - 1^2 = 0
        cr = self._make_composite([0.5], min_output_tokens=1)
        result = cr(source, generated)
        assert result["quality"] == pytest.approx(0.0, abs=0.01)

    def test_total_is_weighted_sum(self):
        weights = {"evasion": 0.5, "semantic": 0.3, "quality": 0.2}
        cr = self._make_composite([0.6], semantic_sim=0.9, weights=weights, min_output_tokens=1)
        source = "one two three four five six seven"
        generated = "eight nine ten eleven twelve thirteen fourteen"
        result = cr(source, generated)
        expected = 0.5 * 0.4 + 0.3 * 0.9 + 0.2 * max(0, result["quality"])
        assert result["total"] == pytest.approx(expected, abs=1e-6)

    def test_per_detector_breakdown(self):
        cr = self._make_composite([0.8, 0.3])
        result = cr(_LONG_SRC, _LONG_GEN)
        assert len(result["per_detector"]) == 2
        assert "det_0" in result["per_detector"]
        assert "det_1" in result["per_detector"]

    def test_no_detectors(self):
        with pytest.raises(ValueError, match="at least one detector"):
            self._make_composite([])

    def test_short_output_rejected(self):
        cr = self._make_composite([0.2], min_output_tokens=20)
        result = cr(_LONG_SRC, "too short")
        assert result["total"] == 0.0
        assert result["rejected"] is True

    def test_default_weights(self):
        detector_rewards = [DetectorReward(lambda t: 0.5, name="d")]

        class MockSemantic:
            def __call__(self, source, generated):
                return 0.9

        cr = CompositeReward(detector_rewards=detector_rewards, semantic_reward=MockSemantic())
        assert cr.weights == {"evasion": 0.6, "semantic": 0.3, "quality": 0.1}
