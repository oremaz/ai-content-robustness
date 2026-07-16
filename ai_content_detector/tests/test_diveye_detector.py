from __future__ import annotations

import numpy as np
import pytest

from ai_content_detector.detectors.text_detectors import DivEyeDetector


def test_diveye_extracts_all_nine_paper_features():
    features = DivEyeDetector._features_from_surprisal(
        np.asarray([1.0, 2.0, 4.0, 3.0, 7.0, 5.0]), entropy_bins=20,
    )
    assert tuple(features) == (
        "mean", "variance", "skew", "kurtosis", "d1_mean",
        "d1_variance", "d2_variance", "d2_entropy", "d2_autocorrelation",
    )
    assert features["mean"] == pytest.approx(22.0 / 6.0)
    assert features["variance"] == pytest.approx(np.var([1, 2, 4, 3, 7, 5], ddof=1))
    assert features["d1_mean"] == pytest.approx(np.diff([1, 2, 4, 3, 7, 5]).mean())
    assert all(np.isfinite(value) for value in features.values())


def test_constant_surprisal_has_finite_moments():
    features = DivEyeDetector._features_from_surprisal(np.ones(8))
    vector = DivEyeDetector._feature_vector(features)
    assert vector.shape == (9,)
    assert np.isfinite(vector).all()


def test_detector_requires_trained_classifier():
    detector = DivEyeDetector(scoring_model="gpt2", device="cpu")
    detector._available = False
    detector._load = lambda: None
    with pytest.raises(RuntimeError, match="not loaded"):
        detector.detect("This text is long enough to score but has no classifier.")
