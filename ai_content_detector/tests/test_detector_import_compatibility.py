"""Historical detector imports must survive implementation moves."""

import pickle

import numpy as np
from PIL import Image

from ai_content_detector import detectors
from ai_content_detector.detectors import image_detectors, text_detectors


def test_historical_detector_class_paths_and_aliases():
    for module in (text_detectors, image_detectors):
        for name, cls in vars(module).items():
            if not isinstance(cls, type) or name == "DivEyeDetector":
                continue
            assert cls.__module__ == module.__name__
            assert getattr(detectors, name) is cls
            legacy_reference = f"c{module.__name__}\n{name}\n.".encode()
            assert pickle.loads(legacy_reference) is cls
    assert text_detectors.InversionDetector is text_detectors.ParaphraseRoundTripDetector
    assert text_detectors.MarkovCalibratedTextDetector is text_detectors.WindowSmoothedTextDetector


def test_image_conversion_preserves_normalization():
    converted = image_detectors._pil_to_tensor(Image.new("RGB", (20, 20), "white"), size=8)
    expected = (1 - np.asarray(image_detectors.IMAGENET_MEAN)) / np.asarray(image_detectors.IMAGENET_STD)
    assert converted.shape == (1, 3, 8, 8)
    assert converted.dtype == np.float32
    np.testing.assert_allclose(converted[0, :, 0, 0], expected, rtol=1e-6)
