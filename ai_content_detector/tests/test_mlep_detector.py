from __future__ import annotations

import numpy as np
import pytest
import torch

from ai_content_detector.detectors import DetectionResult, MLEPDetector


def test_patch_shuffle_moves_whole_patches_and_is_deterministic():
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    first = MLEPDetector._shuffle_patches(image, patch_size=2, seed=7)
    second = MLEPDetector._shuffle_patches(image, patch_size=2, seed=7)
    assert np.array_equal(first, second)
    for channel in range(3):
        original_patches = {
            tuple(image[y:y + 2, x:x + 2, channel].reshape(-1))
            for y in (0, 2) for x in (0, 2)
        }
        shuffled_patches = {
            tuple(first[y:y + 2, x:x + 2, channel].reshape(-1))
            for y in (0, 2) for x in (0, 2)
        }
        assert shuffled_patches == original_patches


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1, 1, 1, 1], 0.0), ([1, 1, 1, 2], 0.811278), ([1, 1, 2, 2], 1.0),
     ([1, 1, 2, 3], 1.5), ([1, 2, 3, 4], 2.0)],
)
def test_local_entropy_has_the_five_paper_levels(values, expected):
    image = np.asarray(values, dtype=np.uint8).reshape(2, 2, 1)
    result = MLEPDetector._local_entropy_patterns(image)
    assert result.shape == (1, 1, 1)
    assert result.item() == pytest.approx(expected, abs=1e-5)


def test_multiscale_map_has_channel_concatenation():
    detector = MLEPDetector(patch_size=2, scales=(1.0, 0.5, 0.25), image_size=8)
    result = detector.extract_mlep_map(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))
    assert result.shape == (7, 7, 9)


def test_detect_requires_and_uses_cnn_classifier():
    classifier = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(9, 1))
    detector = MLEPDetector(classifier=classifier, image_size=8)
    result = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))
    assert isinstance(result, DetectionResult)
    assert 0.0 <= result.score <= 1.0
