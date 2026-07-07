"""Math guards for the MLEP-style entropy-pattern detector."""

from __future__ import annotations

import numpy as np

from ai_content_detector.detectors import MLEPDetector, DetectionResult


def test_entropy_map_shape_and_constant_entropy():
    det = MLEPDetector(patch_sizes=(8,), image_size=32)
    gray = np.full((32, 32), 0.5, dtype=np.float32)

    emap = det._entropy_map(gray, patch_size=8)

    assert emap.shape == (4, 4)
    assert float(np.max(np.abs(emap))) < 1e-6


def test_noise_has_higher_entropy_than_constant():
    det = MLEPDetector(patch_sizes=(8,), image_size=32)
    rng = np.random.default_rng(0)
    noise = rng.random((32, 32), dtype=np.float32)
    constant = np.full((32, 32), 0.5, dtype=np.float32)

    noise_entropy = det._entropy_map(noise, patch_size=8).mean()
    constant_entropy = det._entropy_map(constant, patch_size=8).mean()

    assert float(noise_entropy) > float(constant_entropy) + 1.0


def test_extract_features_contains_artifact_index():
    det = MLEPDetector(patch_sizes=(8, 16), image_size=32)
    rng = np.random.default_rng(1)
    image = rng.random((32, 32, 3), dtype=np.float32)

    features = det.extract_features(image)

    assert "mlep_artifact_index" in features
    assert "mlep_mean_p8" in features
    assert "mlep_grad_p16" in features
    assert features["mlep_artifact_index"] >= 0.0


def test_detect_returns_detection_result():
    det = MLEPDetector(patch_sizes=(8, 16), image_size=32)
    image = np.zeros((32, 32, 3), dtype=np.float32)

    result = det.detect(image)

    assert isinstance(result, DetectionResult)
    assert 0.0 <= result.score <= 1.0
    assert result.detector_name == det.name
