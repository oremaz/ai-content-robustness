"""Checkpoint-backed image classification adapters."""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

from .ensemble import BaseDetector, DetectionResult
from PIL import Image
from .image_detectors_utils import _pil_to_tensor

logger = logging.getLogger("ai_content_detector.detectors.image_detectors")


class EfficientNetDetector(BaseDetector):
    """EfficientNet-B4 trained on ArtiFact (25 generators)."""

    __module__ = "ai_content_detector.detectors.image_detectors"

    name = "EfficientNet-B4"
    modality = "image"

    def __init__(
        self,
        model_name: str = "binary_efficientnet_b4_ns",
        path_start: str = "bench_imai_artifact",
        device: str = "auto",
    ):
        self._model_name = model_name
        self._path_start = path_start
        self._device_spec = device
        self._model = None
        self._predictor = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from ml_pipeline.pipelines_torch.vision_models import MODEL_REGISTRY
            from ml_pipeline.pipelines_torch.base import SimplePredictor
            from ml_pipeline.utils.utils import load_model_by_name

            device = self._device_spec
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device

            model_cls = MODEL_REGISTRY[self._model_name]
            self._model = load_model_by_name(
                model_cls,
                self._model_name,
                {"num_classes": 2},
                path_start=self._path_start,
            )
            self._model.to(device)
            self._model.eval()
            self._predictor = SimplePredictor(
                self._model, task_type="classification", device=device, batch_size=1
            )
            self._available = True
        except Exception as e:
            logger.warning("EfficientNetDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            self._load()
        return bool(self._available)

    def detect(self, content: Union[Image.Image, np.ndarray]) -> DetectionResult:
        self._load()
        if not self._available:
            raise RuntimeError("EfficientNet checkpoint not available")

        if isinstance(content, Image.Image):
            X = _pil_to_tensor(content, size=200)
        else:
            X = content if content.ndim == 4 else content[np.newaxis]

        probs = self._predictor.predict_proba(X)[0]
        # ArtiFact convention: class 0 = fake, class 1 = real
        ai_score = float(probs[0])

        return DetectionResult(
            score=ai_score,
            label=DetectionResult.label_from_score(ai_score),
            details={"prob_fake": float(probs[0]), "prob_real": float(probs[1])},
        )


class CLIPImageDetector(BaseDetector):
    """CLIP-based classifier trained on ArtiFact."""

    __module__ = "ai_content_detector.detectors.image_detectors"

    name = "CLIP Classifier"
    modality = "image"

    def __init__(
        self,
        model_name: str = "clip_classifier",
        path_start: str = "bench_imai_artifact",
        device: str = "auto",
    ):
        self._model_name = model_name
        self._path_start = path_start
        self._device_spec = device
        self._model = None
        self._predictor = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from ml_pipeline.pipelines_torch.vision_models import MODEL_REGISTRY
            from ml_pipeline.pipelines_torch.base import SimplePredictor
            from ml_pipeline.utils.utils import load_model_by_name

            device = self._device_spec
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device

            model_cls = MODEL_REGISTRY[self._model_name]
            self._model = load_model_by_name(
                model_cls,
                self._model_name,
                {"num_classes": 2},
                path_start=self._path_start,
            )
            self._model.to(device)
            self._model.eval()
            self._predictor = SimplePredictor(
                self._model, task_type="classification", device=device, batch_size=1
            )
            self._available = True
        except Exception as e:
            logger.warning("CLIPImageDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            self._load()
        return bool(self._available)

    def detect(self, content: Union[Image.Image, np.ndarray]) -> DetectionResult:
        self._load()
        if not self._available:
            raise RuntimeError("CLIP classifier checkpoint not available")

        if isinstance(content, Image.Image):
            X = _pil_to_tensor(content, size=200)
        else:
            X = content if content.ndim == 4 else content[np.newaxis]

        probs = self._predictor.predict_proba(X)[0]
        ai_score = float(probs[0])

        return DetectionResult(
            score=ai_score,
            label=DetectionResult.label_from_score(ai_score),
            details={"prob_fake": float(probs[0]), "prob_real": float(probs[1])},
        )


class DINOv2Detector(BaseDetector):
    """DINOv2 ViT-Base trained on ArtiFact."""

    __module__ = "ai_content_detector.detectors.image_detectors"

    name = "DINOv2 ViT-B"
    modality = "image"

    def __init__(
        self,
        model_name: str = "timm_dinov2_vit_base",
        path_start: str = "bench_imai_artifact",
        device: str = "auto",
    ):
        self._model_name = model_name
        self._path_start = path_start
        self._device_spec = device
        self._model = None
        self._predictor = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from ml_pipeline.pipelines_torch.vision_models import MODEL_REGISTRY
            from ml_pipeline.pipelines_torch.base import SimplePredictor
            from ml_pipeline.utils.utils import load_model_by_name

            device = self._device_spec
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device

            model_cls = MODEL_REGISTRY[self._model_name]
            self._model = load_model_by_name(
                model_cls,
                self._model_name,
                {"num_classes": 2},
                path_start=self._path_start,
            )
            self._model.to(device)
            self._model.eval()
            self._predictor = SimplePredictor(
                self._model, task_type="classification", device=device, batch_size=1
            )
            self._available = True
        except Exception as e:
            logger.warning("DINOv2Detector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            self._load()
        return bool(self._available)

    def detect(self, content: Union[Image.Image, np.ndarray]) -> DetectionResult:
        self._load()
        if not self._available:
            raise RuntimeError("DINOv2 checkpoint not available")

        if isinstance(content, Image.Image):
            X = _pil_to_tensor(content, size=200)
        else:
            X = content if content.ndim == 4 else content[np.newaxis]

        probs = self._predictor.predict_proba(X)[0]
        ai_score = float(probs[0])

        return DetectionResult(
            score=ai_score,
            label=DetectionResult.label_from_score(ai_score),
            details={"prob_fake": float(probs[0]), "prob_real": float(probs[1])},
        )

