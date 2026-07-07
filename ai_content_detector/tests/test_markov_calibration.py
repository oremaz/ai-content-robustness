"""CPU-only tests for Markov-informed text score calibration."""

from __future__ import annotations

from ai_content_detector.detectors import MarkovCalibratedTextDetector


def test_markov_calibration_smooths_isolated_low_score():
    raw = [0.9, 0.1, 0.9]

    calibrated = MarkovCalibratedTextDetector.calibrate_scores(
        raw,
        neighbor_weight=1.8,
        initial_discount=1.0,
        iterations=6,
    )

    assert calibrated[1] > raw[1]
    assert calibrated[0] > 0.75
    assert calibrated[2] > 0.75


def test_initial_discount_reduces_first_window_instability():
    raw = [0.99, 0.4, 0.4, 0.4]

    discounted = MarkovCalibratedTextDetector.calibrate_scores(
        raw,
        neighbor_weight=0.5,
        initial_discount=0.2,
        iterations=4,
    )
    undiscounted = MarkovCalibratedTextDetector.calibrate_scores(
        raw,
        neighbor_weight=0.5,
        initial_discount=1.0,
        iterations=4,
    )

    assert discounted[0] < undiscounted[0]


def test_detector_wraps_window_scores_and_returns_details():
    calls = []

    def score_fn(window: str) -> float:
        calls.append(window)
        return 0.8 if "machine" in window else 0.2

    det = MarkovCalibratedTextDetector(
        score_fn=score_fn,
        window_size=8,
        stride=4,
        initial_discount=1.0,
    )
    text = "human prose with varied rhythm " * 4 + "machine regular output " * 8

    result = det.detect(text)

    assert calls
    assert result.details["num_windows"] > 1
    assert len(result.details["raw_window_scores"]) == result.details["num_windows"]
    assert 0.0 <= result.score <= 1.0
