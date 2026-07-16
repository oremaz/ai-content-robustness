"""Tests for DetectionResult, BaseDetector, and EnsembleAggregator."""

from __future__ import annotations

import pytest

from ai_content_detector.detectors.ensemble import (
    BaseDetector,
    DetectionResult,
    EnsembleAggregator,
)
from ai_content_detector.tests.conftest import (
    DummyDetector,
    FailingDetector,
    UnavailableDetector,
)


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------


class TestDetectionResult:
    def test_label_from_score_ai(self):
        assert DetectionResult.label_from_score(0.9) == "ai"
        assert DetectionResult.label_from_score(0.65) == "ai"

    def test_label_from_score_human(self):
        assert DetectionResult.label_from_score(0.1) == "human"
        assert DetectionResult.label_from_score(0.35) == "human"

    def test_label_from_score_uncertain(self):
        assert DetectionResult.label_from_score(0.5) == "uncertain"
        assert DetectionResult.label_from_score(0.4) == "uncertain"
        assert DetectionResult.label_from_score(0.64) == "uncertain"

    def test_label_from_score_boundary(self):
        # Exact boundaries
        assert DetectionResult.label_from_score(0.65) == "ai"
        assert DetectionResult.label_from_score(0.35) == "human"

    def test_label_from_score_custom_thresholds(self):
        assert DetectionResult.label_from_score(0.5, ai_threshold=0.4) == "ai"
        assert DetectionResult.label_from_score(0.5, human_threshold=0.6) == "human"

    def test_dataclass_fields(self):
        r = DetectionResult(score=0.7, label="ai", detector_name="test", details={"key": "val"})
        assert r.score == 0.7
        assert r.label == "ai"
        assert r.detector_name == "test"
        assert r.details == {"key": "val"}

    def test_default_details_empty(self):
        r = DetectionResult(score=0.5, label="uncertain")
        assert r.details == {}
        assert r.detector_name == ""

    def test_unload_clears_extended_heavy_resources(self):
        detector = DummyDetector(0.5, "resource")
        detector._reference_model = object()
        detector._clip_model = object()
        detector._uncond_embed = object()
        detector._available = True
        detector.unload()
        assert detector._reference_model is None
        assert detector._clip_model is None
        assert detector._uncond_embed is None


# ---------------------------------------------------------------------------
# EnsembleAggregator
# ---------------------------------------------------------------------------


class TestEnsembleAggregator:
    def test_weighted_average_uniform(self):
        detectors = [DummyDetector(0.8, "a"), DummyDetector(0.4, "b")]
        ens = EnsembleAggregator(detectors)
        result = ens.detect("test")
        assert result["aggregate_score"] == pytest.approx(0.6, abs=1e-6)

    def test_weighted_average_with_weights(self):
        detectors = [DummyDetector(0.8, "a"), DummyDetector(0.2, "b")]
        ens = EnsembleAggregator(detectors, weights={"a": 3.0, "b": 1.0})
        result = ens.detect("test")
        # weighted: (0.8*3 + 0.2*1) / 4 = 2.6/4 = 0.65
        assert result["aggregate_score"] == pytest.approx(0.65, abs=1e-6)

    def test_majority_vote_ai(self):
        detectors = [
            DummyDetector(0.9, "a"),
            DummyDetector(0.8, "b"),
            DummyDetector(0.3, "c"),
        ]
        ens = EnsembleAggregator(detectors, method="majority_vote")
        result = ens.detect("test")
        # 2/3 vote AI
        assert result["aggregate_score"] == pytest.approx(2 / 3, abs=1e-6)

    def test_majority_vote_human(self):
        detectors = [
            DummyDetector(0.1, "a"),
            DummyDetector(0.2, "b"),
            DummyDetector(0.9, "c"),
        ]
        ens = EnsembleAggregator(detectors, method="majority_vote")
        result = ens.detect("test")
        # 1/3 vote AI
        assert result["aggregate_score"] == pytest.approx(1 / 3, abs=1e-6)

    def test_failing_detector_graceful(self):
        detectors = [DummyDetector(0.8, "good"), FailingDetector()]
        ens = EnsembleAggregator(detectors)
        result = ens.detect("test")
        # Failing detector returns 0.5 error score, but _aggregate filters error labels
        # Valid = [0.8], error = [0.5 with label "error"]
        # Only valid used -> 0.8
        assert "per_detector" in result
        assert len(result["per_detector"]) == 2

    def test_unavailable_detector_filtered(self):
        detectors = [DummyDetector(0.8, "good"), UnavailableDetector()]
        ens = EnsembleAggregator(detectors)
        # Unavailable is filtered out in __init__
        assert len(ens.detectors) == 1
        result = ens.detect("test")
        assert result["aggregate_score"] == pytest.approx(0.8, abs=1e-6)

    def test_empty_detectors(self):
        with pytest.raises(ValueError, match="at least one"):
            EnsembleAggregator([])

    def test_per_detector_results(self):
        detectors = [DummyDetector(0.9, "a"), DummyDetector(0.1, "b")]
        ens = EnsembleAggregator(detectors)
        result = ens.detect("test")
        assert len(result["per_detector"]) == 2
        names = [r.detector_name for r in result["per_detector"]]
        assert "a" in names
        assert "b" in names

    def test_aggregate_label(self):
        detectors = [DummyDetector(0.9, "a")]
        ens = EnsembleAggregator(detectors)
        result = ens.detect("test")
        assert result["aggregate_label"] == "ai"
        assert result["aggregate_score_semantics"] == "uncalibrated_ensemble_score"

    def test_single_detector(self):
        ens = EnsembleAggregator([DummyDetector(0.3, "solo")])
        result = ens.detect("test")
        assert result["aggregate_score"] == pytest.approx(0.3, abs=1e-6)
        assert result["aggregate_label"] == "human"
