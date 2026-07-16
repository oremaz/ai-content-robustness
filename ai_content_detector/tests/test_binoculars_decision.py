from __future__ import annotations

import pytest

from ai_content_detector.detectors.text_detectors import BinocularsDetector


@pytest.mark.parametrize(
    ("raw", "label", "below"),
    [(0.89, "ai", True), (0.9015, "human", False), (0.92, "human", False)],
)
def test_label_matches_raw_paper_threshold(monkeypatch, raw, label, below):
    detector = BinocularsDetector(threshold=0.9015)
    detector._available = True
    detector._load = lambda: None
    monkeypatch.setattr(detector, "_compute_score", lambda texts: [raw])
    result = detector.detect("A text long enough for a Binoculars boundary test.")
    assert result.label == label
    assert result.details["below_threshold"] is below
    assert result.details["paper_decision"] == label


def test_threshold_maps_to_half_without_claiming_calibration(monkeypatch):
    detector = BinocularsDetector(threshold=0.9015)
    detector._available = True
    detector._load = lambda: None
    monkeypatch.setattr(detector, "_compute_score", lambda texts: [0.9015])
    result = detector.detect("A text long enough for a Binoculars boundary test.")
    assert result.score == pytest.approx(0.5)
    assert result.details["score_semantics"] == "uncalibrated_monotonic_transform"
