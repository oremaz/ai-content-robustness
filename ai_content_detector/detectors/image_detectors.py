"""Image-based AI content detectors.

Includes internal (checkpoint-based) and external (HuggingFace) detectors.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

from .ensemble import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

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
            # Lightweight check — don't download model yet
            try:
                from transformers import pipeline  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
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
# 6. On-Manifold Patch Classification (OMAT inspired / NeurIPS 2025)
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
# 7. WaRPAD — Faithful reproduction (Choi et al., NeurIPS 2025; arXiv 2511.14030)
# ===========================================================================


class WaRPADDetector(BaseDetector):
    """WaRPAD: training-free detection via cropping robustness.

    Faithful reproduction of the published algorithm (Choi et al., NeurIPS 2025,
    arXiv 2511.14030):

    HFwav(x)  = cos( f(x), f(x − α · HF_haar2(x)) )         α = 0.1
    WaRPAD(x) = (1/n_patch) · Σ_p HFwav(x_p)

    where ``f`` is a self-supervised vision encoder (default: DINOv3 ViT-L/16,
    `facebook/dinov3-vitl16-pretrain-lvd1689m`, CLS token, L2-normalized),
    ``HF_haar2`` extracts the high-frequency content via a 2-level Haar DWT
    (zero out LL2 and reconstruct), and the image is rescaled to ``d_rescale``
    (default 896) and tiled into non-overlapping ``d_patch`` × ``d_patch``
    patches (default 224 — DINOv3 ViT-L/16 native input size).

    The paper used DINOv2 ViT-L/14; we default to DINOv3 ViT-L/16 (released
    Aug 2025) which uses the same parameter scale (~300 M) but is the current
    state-of-the-art self-supervised backbone — note that DINOv3 weights are
    gated on HuggingFace; authenticate with ``huggingface-cli login`` first.

    Generated images yield embeddings that are NOT robust to HF perturbations,
    so HFwav is small for AI and close to 1 for real photos. The reported AI
    score is therefore ``1 − HFwav_avg`` clipped to ``[0, 1]``.
    """

    name = "WaRPAD"
    modality = "image"

    def __init__(
        self,
        backbone: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        d_rescale: int = 896,
        d_patch: int = 224,
        alpha: float = 0.1,
        device: str = "auto",
    ):
        self._backbone_name = backbone
        self._d_rescale = int(d_rescale)
        self._d_patch = int(d_patch)
        self._alpha = float(alpha)
        self._device_spec = device
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
                self._available = True
            except ImportError:
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
        ai_score = float(np.clip(1.0 - hfwav_avg, 0.0, 1.0))

        return DetectionResult(
            score=ai_score,
            label=DetectionResult.label_from_score(ai_score),
            detector_name=self.name,
            details={
                "hfwav_avg": hfwav_avg,
                "n_patch": int(len(orig_patches)),
                "alpha": self._alpha,
                "backbone": self._backbone_name,
            },
        )


# ===========================================================================
# 8. Denoising-Trajectory Detector (DTAD-style; LATTE-inspired implementation)
# ===========================================================================


class DenoisingTrajectoryDetector(BaseDetector):
    """Zero-shot AI-image detector using denoising-trajectory biases.

    Faithful to the framing of DTAD (Liu et al., NeurIPS 2025: "Denoising
    Trajectory Biases for Zero-Shot AI-Generated Image Detection") and built on
    the public LATTE pipeline (Vasilcoiu et al.,
    https://github.com/AnaMVasilcoiu/LATTE-Diffusion-Detector). At each of a
    selected set of timesteps t we add Gaussian noise to the VAE latent of the
    input image and run a single-step DDIM denoise; we then measure the cosine
    similarity between the recovered latent and the original. Empirically,
    diffusion-generated images converge faster along this trajectory (their
    latents lie closer to the model's own manifold), so high mean similarity →
    AI-generated.

    Parameters
    ----------
    sd_model_id
        Pretrained Stable Diffusion identifier (must expose UNet, VAE,
        scheduler — any v1.x checkpoint works).
    timesteps
        Trajectory points to sample. Defaults to a 5-point grid covering
        early/mid/late denoising.
    seed
        RNG seed for the per-timestep noise (kept fixed so the metric is
        deterministic per image).
    """

    name = "Denoising Trajectory"
    modality = "image"

    def __init__(
        self,
        sd_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        timesteps: tuple = (50, 250, 500, 750, 950),
        seed: int = 0,
        device: str = "auto",
    ):
        self._sd_model_id = sd_model_id
        self._timesteps = tuple(int(t) for t in timesteps)
        self._seed = int(seed)
        self._device_spec = device
        self._pipeline = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            import torch
            from diffusers import StableDiffusionPipeline, DDIMScheduler

            device = self._device_spec
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device

            dtype = torch.float16 if device == "cuda" else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                self._sd_model_id, torch_dtype=dtype, safety_checker=None,
            ).to(device)
            # DDIM gives the deterministic single-step prediction we need.
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.set_progress_bar_config(disable=True)
            pipe.unet.eval()
            pipe.vae.eval()
            self._pipeline = pipe

            # Pre-compute the unconditional text embedding once (we use it as
            # the "no prompt" condition for every step).
            with torch.no_grad():
                tokens = pipe.tokenizer(
                    "", padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    return_tensors="pt",
                ).to(device)
                self._uncond_embed = pipe.text_encoder(**tokens).last_hidden_state

            self._available = True
        except Exception as e:
            logger.warning("DenoisingTrajectoryDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import torch  # noqa: F401
                from diffusers import StableDiffusionPipeline  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return bool(self._available)

    def _to_latent(self, image: Image.Image) -> "torch.Tensor":
        import torch
        from torchvision import transforms

        # Stable Diffusion v1.x expects 512×512 inputs to its VAE.
        tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # to [-1, 1]
        ])
        x = tf(image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            dist = self._pipeline.vae.encode(
                x.to(dtype=self._pipeline.vae.dtype)
            ).latent_dist
            latent = dist.mean * 0.18215  # SD scaling factor
        return latent

    def detect(self, content: Union[Image.Image, np.ndarray]) -> DetectionResult:
        import torch
        import torch.nn.functional as F

        self._load()
        if not self._available:
            raise RuntimeError("Denoising trajectory pipeline not available")

        # Normalize input
        if isinstance(content, np.ndarray):
            arr = content
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr)
        else:
            image = content

        latent = self._to_latent(image)            # (1, 4, 64, 64)
        scheduler = self._pipeline.scheduler
        scheduler.set_timesteps(1000, device=self._device)
        all_t = scheduler.timesteps                # descending order

        gen = torch.Generator(device=self._device).manual_seed(self._seed)
        sims: List[float] = []
        for t_idx in self._timesteps:
            # Map our trajectory index (0..999) to the actual scheduler timestep tensor.
            t = all_t[max(0, min(len(all_t) - 1, len(all_t) - 1 - t_idx))]

            noise = torch.randn(
                latent.shape, generator=gen, device=self._device, dtype=latent.dtype,
            )
            noisy = scheduler.add_noise(latent, noise, t.unsqueeze(0))

            with torch.no_grad():
                pred_noise = self._pipeline.unet(
                    noisy, t,
                    encoder_hidden_states=self._uncond_embed.to(latent.dtype),
                ).sample
                # Single-step DDIM update: predicted x0 from epsilon-prediction.
                alpha_bar = scheduler.alphas_cumprod.to(self._device)[t]
                sqrt_ab = alpha_bar.sqrt()
                sqrt_1mab = (1 - alpha_bar).sqrt()
                x0_pred = (noisy - sqrt_1mab * pred_noise) / sqrt_ab.clamp(min=1e-6)

            sim = F.cosine_similarity(
                x0_pred.flatten(1), latent.flatten(1), dim=-1,
            ).item()
            sims.append(float(sim))

        mean_sim = float(np.mean(sims))
        # Generated images converge faster → high cosine similarity → high AI score.
        ai_score = float(np.clip((mean_sim + 1.0) / 2.0, 0.0, 1.0))

        return DetectionResult(
            score=ai_score,
            label=DetectionResult.label_from_score(ai_score),
            detector_name=self.name,
            details={
                "mean_cosine": mean_sim,
                "per_step_cosine": sims,
                "timesteps": list(self._timesteps),
                "sd_model_id": self._sd_model_id,
            },
        )
