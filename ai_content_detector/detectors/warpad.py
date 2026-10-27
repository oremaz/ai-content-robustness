"""WaRPAD self-supervised feature sensitivity detector."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Union

import numpy as np

from .ensemble import BaseDetector, DetectionResult
from PIL import Image
from .image_detectors_utils import _check_hf_image_config

logger = logging.getLogger("ai_content_detector.detectors.image_detectors")


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

    __module__ = "ai_content_detector.detectors.image_detectors"

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

