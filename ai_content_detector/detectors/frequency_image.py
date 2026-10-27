"""Wavelet, entropy and patchwise image artifact detectors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from .image_detectors_utils import IMAGENET_MEAN, IMAGENET_STD
from .ensemble import BaseDetector, DetectionResult
from PIL import Image

logger = logging.getLogger("ai_content_detector.detectors.image_detectors")


class FrequencyDetector(BaseDetector):
    """Lightweight HF energy-ratio heuristic (NOT the published WaRPAD).

    Computes the ratio of high-frequency to low-frequency Haar sub-band energy
    on a single-level DWT and squashes it through a sigmoid. This is a useful
    cheap baseline but is *not* a reproduction of WaRPAD (Choi et al., NeurIPS
    2025), which measures the cosine sensitivity of self-supervised embeddings
    to HF perturbations across patches. For the faithful WaRPAD algorithm see
    ``warpad.WaRPADDetector``.
    """

    __module__ = "ai_content_detector.detectors.image_detectors"

    name = "Wavelet Artifact Detector"
    modality = "image"

    def __init__(self, threshold_energy: float = 0.15):
        self._threshold_energy = threshold_energy
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import pywt  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return bool(self._available)

    def detect(self, content: Union[Image.Image, np.ndarray]) -> DetectionResult:
        if not self.is_available():
            raise RuntimeError("pywavelets not installed")

        import pywt

        if isinstance(content, Image.Image):
            # Convert to grayscale numpy array
            img_arr = np.array(content.convert("L"), dtype=np.float32) / 255.0
        else:
            # Assume ndarray is C,H,W or H,W,C. Convert to 2D grayscale.
            if content.ndim == 4:
                content = content[0]
            if content.ndim == 3:
                # If C,H,W (like 3, 200, 200)
                if content.shape[0] == 3:
                    img_arr = np.mean(content, axis=0)
                else:
                    img_arr = np.mean(content, axis=-1)
            else:
                img_arr = content

        # 2D Discrete Wavelet Transform
        # LL is low freq (approximation), (LH, HL, HH) are high freq (details)
        coeffs2 = pywt.dwt2(img_arr, 'haar')
        LL, (LH, HL, HH) = coeffs2

        # Compute energy of each subband
        energy_ll = np.sum(LL ** 2) + 1e-6
        energy_lh = np.sum(LH ** 2)
        energy_hl = np.sum(HL ** 2)
        energy_hh = np.sum(HH ** 2)

        total_hf_energy = energy_lh + energy_hl + energy_hh
        hf_energy_ratio = float(total_hf_energy / energy_ll)

        # AI images tend to have abnormal high-frequency energy due to upsampling/denoising
        # Map to a score around the threshold. Higher ratio = more likely AI.
        import math
        ai_score = 1.0 / (1.0 + math.exp(-20.0 * (hf_energy_ratio - self._threshold_energy)))

        return DetectionResult(
            score=ai_score,
            label=DetectionResult.label_from_score(ai_score),
            details={
                "hf_energy_ratio": hf_energy_ratio,
                "energy_lh": float(energy_lh),
                "energy_hl": float(energy_hl),
                "energy_hh": float(energy_hh)
            },
        )


class MLEPDetector(BaseDetector):
    """Multi-granularity Local Entropy Patterns with a CNN classifier."""

    __module__ = "ai_content_detector.detectors.image_detectors"

    name = "MLEP Entropy Patterns"
    modality = "image"

    def __init__(
        self,
        patch_size: int = 2,
        scales: Sequence[float] = (1.0, 0.5, 0.25),
        classifier_path: Optional[str] = None,
        classifier: Any = None,
        image_size: int = 256,
        seed: int = 0,
        device: str = "auto",
    ):
        self.patch_size = int(patch_size)
        self.scales = tuple(float(scale) for scale in scales)
        if self.patch_size < 1:
            raise ValueError("patch_size must be positive")
        if not self.scales or any(not 0.0 < scale <= 1.0 for scale in self.scales):
            raise ValueError("scales must be a nonempty sequence in (0, 1]")
        self.classifier_path = classifier_path
        self.image_size = int(image_size)
        self.seed = int(seed)
        self._device_spec = device
        self._classifier = classifier
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._classifier is not None or bool(
                self.classifier_path and Path(self.classifier_path).exists()
            )
        return bool(self._available)

    def _load_classifier(self):
        if self._classifier is not None:
            return
        if not self.classifier_path:
            raise RuntimeError("MLEP requires a trained CNN classifier checkpoint")
        import torch

        device = self._device_spec
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        channels = 3 * len(self.scales)
        self._classifier = self.build_resnet50(channels)
        state = torch.load(self.classifier_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._classifier.load_state_dict(state)
        self._classifier.to(device).eval()

    @staticmethod
    def build_resnet50(input_channels: int):
        import torch.nn as nn
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        model.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,
        )
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    @staticmethod
    def _to_rgb_float(content: Union[Image.Image, np.ndarray], size: int) -> np.ndarray:
        if isinstance(content, Image.Image):
            image = content.convert("RGB")
        else:
            arr = np.asarray(content)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            arr = arr.astype(np.float32)
            arr = arr / 255.0 if arr.max() > 1.5 else arr
            image = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
        image = image.resize((size, size), Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def _shuffle_patches(image: np.ndarray, patch_size: int, seed: int) -> np.ndarray:
        """Spatially permute L by L patches independently for each channel."""
        h, w, channels = image.shape
        h_use = h - h % patch_size
        w_use = w - w % patch_size
        image = image[:h_use, :w_use]
        n_y, n_x = h_use // patch_size, w_use // patch_size
        output = np.empty_like(image)
        for channel in range(channels):
            plane = image[..., channel]
            patches = plane.reshape(n_y, patch_size, n_x, patch_size).transpose(0, 2, 1, 3)
            flat = patches.reshape(n_y * n_x, patch_size, patch_size)
            permutation = np.random.default_rng(seed + channel).permutation(len(flat))
            shuffled = flat[permutation].reshape(n_y, n_x, patch_size, patch_size)
            output[..., channel] = shuffled.transpose(0, 2, 1, 3).reshape(h_use, w_use)
        return output

    @staticmethod
    def _resample(image: np.ndarray, scale: float) -> np.ndarray:
        h, w = image.shape[:2]
        pil = Image.fromarray(image)
        down = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        return np.asarray(down.resize((w, h), Image.BILINEAR), dtype=np.uint8)

    @staticmethod
    def _local_entropy_patterns(image: np.ndarray) -> np.ndarray:
        """Compute exact categorical entropy in every stride-1 2 by 2 window."""
        values = np.stack(
            [image[:-1, :-1], image[1:, :-1], image[:-1, 1:], image[1:, 1:]],
            axis=-1,
        )
        counts = (values[..., :, None] == values[..., None, :]).sum(axis=-1)
        return (-np.log2(counts.astype(np.float32) / 4.0)).mean(axis=-1)

    def extract_mlep_map(self, content: Union[Image.Image, np.ndarray]) -> np.ndarray:
        image = self._to_rgb_float(content, self.image_size)
        shuffled = self._shuffle_patches(image, self.patch_size, self.seed)
        pyramid = np.concatenate(
            [self._resample(shuffled, scale) for scale in self.scales], axis=-1,
        )
        return self._local_entropy_patterns(pyramid).astype(np.float32)

    def detect(self, content: Union[Image.Image, np.ndarray]) -> DetectionResult:
        if not self.is_available():
            raise RuntimeError("MLEP classifier path is not available")

        import torch

        self._load_classifier()
        device = next(self._classifier.parameters()).device
        feature_map = self.extract_mlep_map(content)
        tensor = torch.from_numpy(feature_map.transpose(2, 0, 1)).unsqueeze(0).to(device)
        tensor = tensor / 2.0
        self._classifier.eval()
        with torch.no_grad():
            logit = self._classifier(tensor).reshape(-1)[0]
        ai_score = float(torch.sigmoid(logit).item())
        return DetectionResult(
            score=ai_score,
            label=DetectionResult.label_from_score(ai_score),
            detector_name=self.name,
            details={
                "mlep_shape": list(feature_map.shape),
                "patch_size": self.patch_size,
                "scales": list(self.scales),
                "classifier_path": self.classifier_path,
                "score_semantics": "classifier_probability",
            },
        )


class PatchBasedClassifier(BaseDetector):
    """Wraps an existing detector to perform patch-based evaluation.

    Forces the model to look for local generative artifacts instead of
    global spurious correlations (like lighting or watermarks), drastically
    improving cross-generator generalization and robustness.
    """

    __module__ = "ai_content_detector.detectors.image_detectors"

    name = "Patch-Based Classifier"
    modality = "image"

    def __init__(self, base_detector: BaseDetector, patch_size: int = 128, stride: int = 64):
        self.base_detector = base_detector
        self.name = f"Patch-Based {base_detector.name}"
        self.patch_size = patch_size
        self.stride = stride

    def is_available(self) -> bool:
        return self.base_detector.is_available()

    def unload(self) -> None:
        """Delegate to the wrapped detector, which holds the actual model."""
        self.base_detector.unload()

    def detect(self, content: Union[Image.Image, np.ndarray]) -> DetectionResult:
        if not self.is_available():
            raise RuntimeError(f"{self.base_detector.name} not available")

        # Convert to PIL Image for easy cropping if it's an array
        if isinstance(content, np.ndarray):
            if content.ndim == 4:
                content = content[0]
            if content.ndim == 3 and content.shape[0] == 3:
                # C,H,W to H,W,C
                content = np.transpose(content, (1, 2, 0))
                # Unnormalize if necessary (heuristic)
                if content.min() < 0:
                    content = (content * IMAGENET_STD) + IMAGENET_MEAN
                    content = np.clip(content, 0, 1)

            if content.max() <= 1.0:
                content = (content * 255).astype(np.uint8)
            else:
                content = content.astype(np.uint8)
            img = Image.fromarray(content)
        else:
            img = content.convert("RGB")

        w, h = img.size
        patch_scores = []

        # If image is smaller than patch, just score the whole thing
        if w <= self.patch_size or h <= self.patch_size:
            return self.base_detector.detect(img)

        # Sliding window
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
                res = self.base_detector.detect(patch)
                patch_scores.append(res.score)

        if not patch_scores:
            return self.base_detector.detect(img)

        # Aggregate patch scores. We use a high percentile to detect if ANY
        # significant part of the image has deepfake artifacts.
        agg_score = float(np.percentile(patch_scores, 85)) # 85th percentile

        return DetectionResult(
            score=agg_score,
            label=DetectionResult.label_from_score(agg_score),
            details={
                "mean_patch_score": float(np.mean(patch_scores)),
                "max_patch_score": float(np.max(patch_scores)),
                "num_patches": len(patch_scores)
            },
        )

