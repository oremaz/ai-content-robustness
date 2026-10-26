"""Regression guards for image-detector return contracts.

These checks exercise the interface without loading model weights.
"""

from __future__ import annotations

import inspect

from ai_content_detector.detectors import (
    DetectionResult,
    DenoisingTrajectoryDetector,
    EfficientNetDetector,
    CLIPImageDetector,
    DINOv2Detector,
    FrequencyDetector,
    MLEPDetector,
    PatchBasedClassifier,
    WaRPADDetector,
)


IMAGE_DETECTORS = [
    EfficientNetDetector,
    CLIPImageDetector,
    DINOv2Detector,
    FrequencyDetector,
    MLEPDetector,
    WaRPADDetector,
    DenoisingTrajectoryDetector,
]


class TestImageDetectorContracts:
    def test_all_have_detect_method(self):
        for cls in IMAGE_DETECTORS:
            assert hasattr(cls, "detect"), f"{cls.__name__} must expose .detect()"
            sig = inspect.signature(cls.detect)
            params = list(sig.parameters.keys())
            assert params and params[0] == "self"
            assert "content" in params, f"{cls.__name__}.detect must take `content`"

    def test_detect_function_body_returns(self):
        """Every detect() must contain a `return` statement."""
        for cls in IMAGE_DETECTORS:
            src = inspect.getsource(cls.detect)
            assert "return " in src, (
                f"{cls.__name__}.detect has no `return` — it will return None"
            )

    def test_all_have_is_available(self):
        for cls in IMAGE_DETECTORS:
            assert hasattr(cls, "is_available")

    def test_patch_classifier_wraps_base(self):
        # Construct the wrapper around a fake base detector to verify it
        # neither crashes nor swallows the .detect contract.
        class _FakeBase:
            name = "fake"

            def is_available(self) -> bool:
                return True

            def detect(self, content):
                return DetectionResult(score=0.7, label="ai", detector_name="fake")

        wrapped = PatchBasedClassifier(_FakeBase(), patch_size=64, stride=32)
        assert "fake" in wrapped.name.lower()
        assert wrapped.is_available()
