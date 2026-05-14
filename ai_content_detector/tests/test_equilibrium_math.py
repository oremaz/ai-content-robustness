"""Math-correctness tests for the arms-race Nash gap.

The previous implementation used ``defender_AUROC - mean_evasion`` which is
not bounded in ``[0, 1]`` and has no game-theoretic interpretation. The
fixed version uses the best-response deficit
``max(0, defender_accuracy - (1 - attacker_success_rate))`` clipped to
``[0, 1]``.
"""

from __future__ import annotations

import pytest

from ai_content_detector.rl_evasion.arms_race.equilibrium import ArmsRaceExperiment


def _make_round(attacker_success: float, defender_accuracy: float, **extra) -> dict:
    return {
        "round": 1,
        "attacker": {"attack_success_rate": attacker_success, **extra.get("attacker", {})},
        "defender": {"accuracy": defender_accuracy, **extra.get("defender", {})},
    }


class TestNashGapBounds:
    @pytest.mark.parametrize("atk", [0.0, 0.25, 0.5, 0.75, 1.0])
    @pytest.mark.parametrize("dac", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_gap_in_unit_interval(self, atk: float, dac: float):
        gap = ArmsRaceExperiment._compute_nash_gap(_make_round(atk, dac))
        assert 0.0 <= gap <= 1.0

    def test_perfect_defender_zero_attacker_yields_zero_gap(self):
        # Defender already catches every AI sample the attacker fails to bypass
        # (which is all of them, since attacker_success=0). No further deficit
        # to close → gap = 0. The "max gap" intuition belongs to the dual case
        # where attacker bypasses a lot AND defender has high accuracy on the
        # leftovers.
        gap = ArmsRaceExperiment._compute_nash_gap(_make_round(0.0, 1.0))
        assert gap == pytest.approx(0.0)

    def test_max_gap_when_both_sides_strong(self):
        # gap = max(0, defender_acc - (1 - attacker_success)) = max(0, 1 + 1 - 1) = 1.
        # Mathematically possible only at the corners; physically rare but a useful
        # bound to lock down.
        gap = ArmsRaceExperiment._compute_nash_gap(_make_round(1.0, 1.0))
        assert gap == pytest.approx(1.0)

    def test_full_attacker_partial_defender(self):
        # Attacker bypasses 100% but defender still catches 50% overall →
        # gap = max(0, 0.5 - 0) = 0.5.
        gap = ArmsRaceExperiment._compute_nash_gap(_make_round(1.0, 0.5))
        assert gap == pytest.approx(0.5)

    def test_full_attacker_zero_defender(self):
        # Defender catches nothing → gap = max(0, 0 - 0) = 0.
        gap = ArmsRaceExperiment._compute_nash_gap(_make_round(1.0, 0.0))
        assert gap == pytest.approx(0.0)

    def test_gap_is_zero_when_defender_already_beaten(self):
        # Defender accuracy ≤ what the attacker has already not-bypassed.
        gap = ArmsRaceExperiment._compute_nash_gap(_make_round(0.5, 0.4))
        # 0.4 - (1 - 0.5) = 0.4 - 0.5 = -0.1 → clamped to 0.
        assert gap == pytest.approx(0.0)


class TestNashGapMonotonicity:
    def test_gap_grows_with_defender_accuracy(self):
        atk = 0.4
        gaps = [
            ArmsRaceExperiment._compute_nash_gap(_make_round(atk, dac))
            for dac in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        # Non-decreasing in defender accuracy
        assert all(b >= a for a, b in zip(gaps, gaps[1:]))

    def test_gap_grows_with_attacker_success(self):
        # When the defender's accuracy is fixed, an attacker that bypasses MORE
        # leaves less room for the defender to improve, so the gap shrinks.
        # Wait — re-derive: gap = max(0, dac - (1 - atk)) = max(0, dac + atk - 1).
        # So increasing atk should *increase* the gap (until clamp). Locking that.
        dac = 0.6
        gaps = [
            ArmsRaceExperiment._compute_nash_gap(_make_round(atk, dac))
            for atk in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ]
        assert all(b >= a for a, b in zip(gaps, gaps[1:]))


class TestNashGapEmptyDefender:
    def test_handles_missing_defender(self):
        round_ = {"round": 1, "attacker": {"attack_success_rate": 0.5}, "defender": {}}
        gap = ArmsRaceExperiment._compute_nash_gap(round_)
        # Defender accuracy defaults to 0 → max(0, 0 - 0.5) = 0.
        assert gap == pytest.approx(0.0)

    def test_handles_none_defender(self):
        round_ = {"round": 1, "attacker": {"attack_success_rate": 0.7}, "defender": None}
        gap = ArmsRaceExperiment._compute_nash_gap(round_)
        assert gap == pytest.approx(0.0)
