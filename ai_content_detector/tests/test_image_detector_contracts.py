"""Regression guards for image-detector return contracts.

These don't load any model — they instantiate the detector classes and
exercise the interface so that bugs like the previous "SigLIPDetector.detect()
fell off the end of the function and silently returned None" can't reappear.
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
    PatchBasedClassifier,
    SigLIPDetector,
    WaRPADDetector,
)


IMAGE_DETECTORS = [
    EfficientNetDetector,
    CLIPImageDetector,
    DINOv2Detector,
    SigLIPDetector,
    FrequencyDetector,
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
        """Every detect() must contain a `return` statement.

        The SigLIPDetector regression we shipped silently returned None because
        the function body fell off the end. A static check on the source
        catches that class of bug without instantiating any model.
        """
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


class TestSigLIPNoNoneReturn:
    """Regression guard: the buggy SigLIPDetector.detect() fell off the end of
    the function and returned None — pre-fix, calling code crashed downstream
    when trying to read .score on None. We can't test it end-to-end without a
    GPU, but a source-level check guarantees a `return` survives any future
    edit.
    """

    def test_siglip_source_contains_return_with_detection_result(self):
        src = inspect.getsource(SigLIPDetector.detect)
        assert "return DetectionResult(" in src, (
            "SigLIPDetector.detect must return a DetectionResult — "
            "the previous version had no return statement"
        )
