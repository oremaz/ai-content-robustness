from .text_detectors import (
    ModernBERTDetector,
    TFIDFDetector,
    BinocularsDetector,
    FastDetectGPTDetector,
    DivEyeDetector,
    InversionDetector,  # alias kept for back-compat
    ParaphraseRoundTripDetector,
    IPADDetector,
)
from .image_detectors import (
    EfficientNetDetector,
    CLIPImageDetector,
    DINOv2Detector,
    SigLIPDetector,
    FrequencyDetector,
    PatchBasedClassifier,
    WaRPADDetector,
    DenoisingTrajectoryDetector,
)
from .style_detector import StyleEmbeddingDetector
from .ensemble import EnsembleAggregator, DetectionResult, BaseDetector

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "EnsembleAggregator",
    "ModernBERTDetector",
    "TFIDFDetector",
    "BinocularsDetector",
    "FastDetectGPTDetector",
    "DivEyeDetector",
    "InversionDetector",
    "ParaphraseRoundTripDetector",
    "IPADDetector",
    "EfficientNetDetector",
    "CLIPImageDetector",
    "DINOv2Detector",
    "SigLIPDetector",
    "FrequencyDetector",
    "PatchBasedClassifier",
    "WaRPADDetector",
    "DenoisingTrajectoryDetector",
    "StyleEmbeddingDetector",
]
