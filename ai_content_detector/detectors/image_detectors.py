"""Image-based AI content detectors.

Includes internal (checkpoint-based) and external (HuggingFace) detectors.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from .ensemble import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


def _check_hf_image_config(instance, model_name: str) -> bool:
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_name)
        return True
    except Exception as error:
        instance._availability_error = str(error)
        return False

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ML_PIPELINE = _REPO_ROOT / "ml_pipeline"
if str(_ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_ML_PIPELINE))

# ImageNet normalization constants (shared across vision detectors)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pil_to_tensor(image: Image.Image, size: int = 200) -> "np.ndarray":
    """Convert a PIL image to a normalized tensor (C, H, W) as float32 numpy."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(image.convert("RGB"))
    return tensor.unsqueeze(0).numpy().astype(np.float32)


# ===========================================================================
# 1. EfficientNet-B4-NS (from bench-imai-artifact)
# ===========================================================================


class EfficientNetDetector(BaseDetector):
    """EfficientNet-B4 trained on ArtiFact (25 generators)."""

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
            from pipelines_torch.vision_models import MODEL_REGISTRY
            from pipelines_torch.base import SimplePredictor
            from utils.utils import load_model_by_name

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


# ===========================================================================
# 2. CLIP classifier (from bench-imai-artifact)
# ===========================================================================


class CLIPImageDetector(BaseDetector):
    """CLIP-based classifier trained on ArtiFact — best cross-dataset generalizer."""

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
            from pipelines_torch.vision_models import MODEL_REGISTRY
            from pipelines_torch.base import SimplePredictor
            from utils.utils import load_model_by_name

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


# ===========================================================================
# 3. DINOv2-ViT-B (from bench-imai-artifact)
# ===========================================================================


class DINOv2Detector(BaseDetector):
    """DINOv2 ViT-Base trained on ArtiFact."""

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
            from pipelines_torch.vision_models import MODEL_REGISTRY
            from pipelines_torch.base import SimplePredictor
            from utils.utils import load_model_by_name

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


# ===========================================================================
# 4. SigLIP-2 deepfake detector (HuggingFace)
# ===========================================================================


class SigLIPDetector(BaseDetector):
    """SigLIP-2 based deepfake/AI-image detector from HuggingFace.

    Uses a fine-tuned SigLIP-2 vision transformer for binary classification.
    """

    name = "SigLIP-2 Detector"
    modality = "image"

    def __init__(
        self,
        model_id: str = "prithivMLmods/deepfake-detector-model-v1",
        device: str = "auto",
    ):
        self._model_id = model_id
        self._device_spec = device
        self._pipe = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from transformers import pipeline

            device_idx = 0 if (self._device_spec == "auto" and torch.cuda.is_available()) else -1
            if self._device_spec not in ("auto", "cpu", "cuda"):
                device_idx = int(self._device_spec.split(":")[-1]) if ":" in self._device_spec else -1

            self._pipe = pipeline(
                "image-classification",
                model=self._model_id,
                device=device_idx,
            )
            self._available = True
        except Exception as e:
            logger.warning("SigLIPDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            self._available = _check_hf_image_config(self, self._model_id)
        return bool(self._available)

    def detect(self, content: Union[Image.Image, np.ndarray]) -> DetectionResult:
        self._load()
        if not self._available:
            raise RuntimeError("SigLIP pipeline not available")

        if isinstance(content, np.ndarray):
            content = Image.fromarray(
                (content * 255).astype(np.uint8) if content.max() <= 1.0 else content.astype(np.uint8)
            )

        results = self._pipe(content)

        # Parse results — format: [{"label": "Real/Fake", "score": 0.99}, ...]
        label_scores = {r["label"].lower(): r["score"] for r in results}
        ai_score = float(
            label_scores.get("fake", label_scores.get("ai", label_scores.get("synthetic", 0.5)))
        )
        return DetectionResult(
            score=ai_score,
            label="ai" if ai_score >= 0.5 else "human",
            detector_name=self.name,
            details={"raw": label_scores},
        )


# ===========================================================================
# 5. High-Frequency Wavelet Analysis (WaRPAD, NeurIPS 2025)
# ===========================================================================


class FrequencyDetector(BaseDetector):
    """Lightweight HF energy-ratio heuristic (NOT the published WaRPAD).

    Computes the ratio of high-frequency to low-frequency Haar sub-band energy
    on a single-level DWT and squashes it through a sigmoid. This is a useful
    cheap baseline but is *not* a reproduction of WaRPAD (Choi et al., NeurIPS
    2025), which measures the cosine sensitivity of self-supervised embeddings
    to HF perturbations across patches. For the faithful WaRPAD algorithm see
    ``WaRPADDetector`` below.
    """

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


# ===========================================================================
# 6. MLEP entropy-pattern detector (NeurIPS 2025)
# ===========================================================================


class MLEPDetector(BaseDetector):
    """Multi-granularity Local Entropy Patterns with a CNN classifier."""

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


# ===========================================================================
# 7. On-Manifold Patch Classification (OMAT inspired / NeurIPS 2025)
# ===========================================================================


class PatchBasedClassifier(BaseDetector):
    """Wraps an existing detector to perform patch-based evaluation.

    Forces the model to look for local generative artifacts instead of
    global spurious correlations (like lighting or watermarks), drastically
    improving cross-generator generalization and robustness.
    """

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


# ===========================================================================
# 8. WaRPAD — Faithful reproduction (Choi et al., NeurIPS 2025; arXiv 2511.14030)
# ===========================================================================


class WaRPADDetector(BaseDetector):
    """WaRPAD: training-free detection via cropping robustness.

    Faithful reproduction of the published algorithm (Choi et al., NeurIPS 2025,
    arXiv 2511.14030):

    HFwav(x)  = cos( f(x), f(x − α · HF_haar2(x)) )         α = 0.1
    WaRPAD(x) = (1/n_patch) · Σ_p HFwav(x_p)

    where ``f`` is the paper's DINOv2 ViT-L/14 encoder, CLS token, L2-normalized,
    ``HF_haar2`` extracts the high-frequency content via a 2-level Haar DWT
    (zero out LL2 and reconstruct), and the image is rescaled to ``d_rescale``
    (default 896) and tiled into non-overlapping ``d_patch`` × ``d_patch``
    patches (default 224).

    Generated images yield embeddings that are NOT robust to HF perturbations,
    so HFwav is small for AI and close to 1 for real photos. The reported AI
    score is therefore ``1 − HFwav_avg`` clipped to ``[0, 1]``.
    """

    name = "WaRPAD"
    modality = "image"

    def __init__(
        self,
        backbone: str = "facebook/dinov2-large",
        d_rescale: int = 896,
        d_patch: int = 224,
        alpha: float = 0.1,
        device: str = "auto",
        threshold: Optional[float] = None,
        calibrator: Optional[Callable[[float], float]] = None,
    ):
        self._backbone_name = backbone
        self._d_rescale = int(d_rescale)
        self._d_patch = int(d_patch)
        self._alpha = float(alpha)
        self._device_spec = device
        self._threshold = threshold
        self._calibrator = calibrator
        self._model = None
        self._processor = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            device = self._device_spec
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device

            self._processor = AutoImageProcessor.from_pretrained(self._backbone_name)
            self._model = AutoModel.from_pretrained(self._backbone_name).to(device)
            self._model.eval()
            self._available = True
        except Exception as e:
            logger.warning("WaRPADDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import pywt  # noqa: F401
                from transformers import AutoModel  # noqa: F401
                self._available = _check_hf_image_config(self, self._backbone_name)
            except Exception as error:
                self._availability_error = str(error)
                self._available = False
        return bool(self._available)

    @staticmethod
    def _high_freq_haar2(arr: np.ndarray) -> np.ndarray:
        """Extract HF content via 2-level Haar DWT: zero LL2, reconstruct.

        Operates per channel on a (H, W, C) float array in [0, 1].
        """
        import pywt

        out = np.zeros_like(arr, dtype=np.float32)
        for c in range(arr.shape[-1]):
            coeffs = pywt.wavedec2(arr[..., c], "haar", level=2)
            # coeffs = [LL2, (LH2,HL2,HH2), (LH1,HL1,HH1)]
            ll2 = coeffs[0]
            coeffs[0] = np.zeros_like(ll2)
            recon = pywt.waverec2(coeffs, "haar")
            # waverec2 may return arr +/- 1 px on odd dims; crop to original shape
            recon = recon[: arr.shape[0], : arr.shape[1]]
            out[..., c] = recon.astype(np.float32)
        return out

    def _embed_batch(self, patches: List[np.ndarray]) -> "np.ndarray":
        """Embed a list of (H, W, 3) float [0,1] patches with the DINO backbone.

        Returns (N, D) L2-normalized CLS-token embeddings.
        """
        import torch
        from PIL import Image as _PIL

        pil = [
            _PIL.fromarray(np.clip(p * 255.0, 0.0, 255.0).astype(np.uint8))
            for p in patches
        ]
        inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
            # DINOv2/v3 last_hidden_state: (B, 1+N_patches, D); CLS = [:, 0]
            cls = out.last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, dim=-1)
        return cls.cpu().numpy()

    def detect(self, content: Union[Image.Image, np.ndarray]) -> DetectionResult:
        self._load()
        if not self._available:
            raise RuntimeError("WaRPAD backbone not available")

        # Normalize input → (H, W, 3) float [0,1]
        if isinstance(content, Image.Image):
            img = np.asarray(content.convert("RGB"), dtype=np.float32) / 255.0
        else:
            arr = np.asarray(content)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
                arr = np.transpose(arr, (1, 2, 0))
            arr = arr.astype(np.float32)
            img = arr / 255.0 if arr.max() > 1.5 else arr

        # Rescale to d_rescale × d_rescale (PIL bilinear; matches paper).
        pil = Image.fromarray(np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8))
        pil = pil.resize((self._d_rescale, self._d_rescale), Image.BILINEAR)
        img = np.asarray(pil, dtype=np.float32) / 255.0

        # HF content + perturbed image
        hf = self._high_freq_haar2(img)
        perturbed = np.clip(img - self._alpha * hf, 0.0, 1.0)

        # Tile into non-overlapping d_patch × d_patch patches
        ps = self._d_patch
        n_side = self._d_rescale // ps
        if n_side < 1:
            raise ValueError(
                f"d_rescale={self._d_rescale} smaller than d_patch={ps}; can't tile"
            )

        orig_patches: List[np.ndarray] = []
        pert_patches: List[np.ndarray] = []
        for i in range(n_side):
            for j in range(n_side):
                y, x = i * ps, j * ps
                orig_patches.append(img[y : y + ps, x : x + ps])
                pert_patches.append(perturbed[y : y + ps, x : x + ps])

        # Embed both sets and compute per-patch cosine similarity
        emb_orig = self._embed_batch(orig_patches)        # (n_patch, D)
        emb_pert = self._embed_batch(pert_patches)        # (n_patch, D)
        hfwav_per_patch = (emb_orig * emb_pert).sum(axis=-1)   # already L2-normalized
        hfwav_avg = float(np.mean(hfwav_per_patch))

        # Generated images are non-robust → low cosine → high AI score.
        raw_ai_score = float(np.clip(1.0 - hfwav_avg, 0.0, 1.0))
        if self._calibrator is not None:
            ai_score = float(self._calibrator(raw_ai_score))
            label = DetectionResult.label_from_score(ai_score, 0.5, 0.5)
            semantics = "calibrated_probability"
        else:
            ai_score = raw_ai_score
            label = "uncertain" if self._threshold is None else (
                "ai" if raw_ai_score >= self._threshold else "human"
            )
            semantics = "raw_warpad_anomaly_score"

        return DetectionResult(
            score=ai_score,
            label=label,
            detector_name=self.name,
            details={
                "hfwav_avg": hfwav_avg,
                "n_patch": int(len(orig_patches)),
                "alpha": self._alpha,
                "backbone": self._backbone_name,
                "raw_warpad_score": raw_ai_score,
                "threshold": self._threshold,
                "score_semantics": semantics,
            },
        )


# ===========================================================================
# 9. Denoising-Trajectory Detector (DTAD-style; LATTE-inspired implementation)
# ===========================================================================


class DenoisingTrajectoryDetector(BaseDetector):
    """DTAD using DDIM inversion and intermediate CLIP image features."""

    name = "Denoising Trajectory"
    modality = "image"

    def __init__(
        self,
        sd_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        clip_model_id: str = "openai/clip-vit-large-patch14",
        num_inference_steps: int = 50,
        trajectory_indices: Optional[Sequence[int]] = None,
        clip_layer: int = 15,
        device: str = "auto",
    ):
        self._sd_model_id = sd_model_id
        self._clip_model_id = clip_model_id
        self._num_inference_steps = int(num_inference_steps)
        self._trajectory_indices = None if trajectory_indices is None else tuple(trajectory_indices)
        self._clip_layer = int(clip_layer)
        self._device_spec = device
        self._pipeline = None
        self._clip_model = None
        self._clip_processor = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._pipeline is not None and self._clip_model is not None:
            return
        try:
            import torch
            from diffusers import DDIMScheduler, StableDiffusionPipeline
            from transformers import AutoProcessor, CLIPVisionModel

            device = self._device_spec
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            dtype = torch.float16 if device == "cuda" else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                self._sd_model_id, torch_dtype=dtype, safety_checker=None,
            ).to(device)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.set_progress_bar_config(disable=True)
            pipe.unet.eval()
            pipe.vae.eval()
            self._pipeline = pipe
            self._clip_processor = AutoProcessor.from_pretrained(self._clip_model_id)
            self._clip_model = CLIPVisionModel.from_pretrained(
                self._clip_model_id, torch_dtype=dtype,
            ).to(device).eval()
            with torch.no_grad():
                tokens = pipe.tokenizer(
                    "", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                    return_tensors="pt",
                ).to(device)
                self._uncond_embed = pipe.text_encoder(**tokens).last_hidden_state
            self._available = True
        except Exception as error:
            logger.warning("DenoisingTrajectoryDetector unavailable: %s", error)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from diffusers import StableDiffusionPipeline
                from transformers import CLIPVisionModel  # noqa: F401
                StableDiffusionPipeline.load_config(self._sd_model_id)
                self._available = _check_hf_image_config(self, self._clip_model_id)
            except Exception as error:
                self._availability_error = str(error)
                self._available = False
        return bool(self._available)

    def _to_latent(self, image: Image.Image) -> "torch.Tensor":
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((512, 512)), transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            distribution = self._pipeline.vae.encode(
                tensor.to(dtype=self._pipeline.vae.dtype),
            ).latent_dist
        return distribution.mean * self._pipeline.vae.config.scaling_factor

    @staticmethod
    def _ddim_inversion_update(latent, predicted_noise, alpha_current, alpha_next):
        predicted_clean = (
            latent - (1.0 - alpha_current).sqrt() * predicted_noise
        ) / alpha_current.sqrt().clamp(min=1e-8)
        next_latent = (
            alpha_next.sqrt() * predicted_clean
            + (1.0 - alpha_next).sqrt() * predicted_noise
        )
        return next_latent, predicted_clean

    def _decode_latents(self, latents: List["torch.Tensor"]) -> List[Image.Image]:
        import torch

        images = []
        scaling = self._pipeline.vae.config.scaling_factor
        for latent in latents:
            with torch.no_grad():
                decoded = self._pipeline.vae.decode(latent / scaling).sample
            array = decoded[0].float().cpu().permute(1, 2, 0).numpy()
            images.append(Image.fromarray(
                np.clip((array / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8),
            ))
        return images

    def _clip_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        import torch

        inputs = self._clip_processor(images=images, return_tensors="pt")
        pixels = inputs["pixel_values"].to(
            self._device, dtype=next(self._clip_model.parameters()).dtype,
        )
        with torch.no_grad():
            outputs = self._clip_model(pixel_values=pixels, output_hidden_states=True)
        if self._clip_layer >= len(outputs.hidden_states):
            raise ValueError(
                f"CLIP layer {self._clip_layer} unavailable; model has {len(outputs.hidden_states)} states"
            )
        embeddings = outputs.hidden_states[self._clip_layer][:, 0].float()
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.cpu().numpy()

    def detect(self, content: Union[Image.Image, np.ndarray]) -> DetectionResult:
        import torch

        self._load()
        if not self._available:
            raise RuntimeError("Denoising trajectory pipeline not available")
        if isinstance(content, np.ndarray):
            array = np.asarray(content)
            if array.ndim == 4:
                array = array[0]
            if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
                array = np.transpose(array, (1, 2, 0))
            if array.dtype != np.uint8:
                array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255).astype(np.uint8)
            image = Image.fromarray(array).convert("RGB")
        else:
            image = content.convert("RGB")

        latent = self._to_latent(image)
        scheduler = self._pipeline.scheduler
        scheduler.set_timesteps(self._num_inference_steps, device=self._device)
        timesteps = list(reversed(scheduler.timesteps))
        selected = set(range(len(timesteps)) if self._trajectory_indices is None else self._trajectory_indices)
        if any(index < 0 or index >= len(timesteps) for index in selected):
            raise ValueError("trajectory_indices contains an out-of-range inversion step")

        alphas = scheduler.alphas_cumprod.to(self._device, dtype=latent.dtype)
        current = latent
        denoising_outputs = []
        selected_timesteps = []
        for index, timestep in enumerate(timesteps):
            with torch.no_grad():
                predicted_noise = self._pipeline.unet(
                    current, timestep,
                    encoder_hidden_states=self._uncond_embed.to(latent.dtype),
                ).sample
            alpha_current = alphas[timestep]
            alpha_next = alphas[timesteps[index + 1]] if index + 1 < len(timesteps) else alpha_current
            current, predicted_clean = self._ddim_inversion_update(
                current, predicted_noise, alpha_current, alpha_next,
            )
            if index in selected:
                denoising_outputs.append(predicted_clean.detach())
                selected_timesteps.append(int(timestep.item()))

        trajectory_images = self._decode_latents(denoising_outputs)
        original = image.resize((512, 512), Image.BILINEAR)
        embeddings = self._clip_embeddings([original] + trajectory_images)
        similarities = embeddings[1:] @ embeddings[0]
        mean_similarity = float(similarities.mean())
        ai_score = float(np.clip((mean_similarity + 1.0) / 2.0, 0.0, 1.0))
        return DetectionResult(
            score=ai_score,
            label=DetectionResult.label_from_score(ai_score),
            detector_name=self.name,
            details={
                "mean_clip_similarity": mean_similarity,
                "summed_clip_similarity": float(similarities.sum()),
                "per_step_clip_similarity": similarities.tolist(),
                "timesteps": selected_timesteps,
                "sd_model_id": self._sd_model_id,
                "clip_model_id": self._clip_model_id,
                "clip_layer": self._clip_layer,
                "score_semantics": "uncalibrated_similarity_transform",
            },
        )
