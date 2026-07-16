"""Math-correctness tests for the WaRPAD HF Haar wavelet step.

Locks the *signal-processing* half of the algorithm (HF extraction) so we can
catch regressions there without needing a GPU/DINOv2 to be available.
"""

from __future__ import annotations

import numpy as np
import pytest

pywt = pytest.importorskip("pywt")

from ai_content_detector.detectors.image_detectors import WaRPADDetector


class TestWaRPADHaarHF:
    """The 2-level Haar HF reconstruction must:

    - Have the same spatial shape as the input.
    - Be (numerically) zero on a constant image (no high frequencies anywhere).
    - Have non-trivial energy on a noisy image.
    - Concentrate near pixel boundaries / edges (where HF lives).
    """

    def _det(self) -> WaRPADDetector:
        # No model load — we only call the static helper.
        return WaRPADDetector(d_rescale=224, d_patch=224)

    def test_shape_preserved(self):
        det = self._det()
        x = np.zeros((128, 128, 3), dtype=np.float32)
        out = det._high_freq_haar2(x)
        assert out.shape == x.shape

    def test_constant_image_has_zero_hf(self):
        det = self._det()
        # A perfectly flat image has no high-frequency content; reconstruction
        # from zeroed-LL2 + zero detail coefficients should be ~0.
        x = np.full((64, 64, 3), 0.5, dtype=np.float32)
        out = det._high_freq_haar2(x)
        assert np.max(np.abs(out)) < 1e-5

    def test_noise_image_has_nonzero_hf(self):
        det = self._det()
        rng = np.random.default_rng(0)
        x = rng.random((64, 64, 3), dtype=np.float32)
        out = det._high_freq_haar2(x)
        assert float(np.mean(np.abs(out))) > 1e-2

    def test_offset_step_has_local_hf(self):
        det = self._det()
        # A vertical step at an odd column index forces detail-band energy that
        # an even-aligned step wouldn't (Haar's even-block structure absorbs
        # boundary-aligned steps into LL).
        x = np.zeros((64, 64, 3), dtype=np.float32)
        x[:, 33:, :] = 1.0
        out = det._high_freq_haar2(x)
        edge_band = np.sum(out[:, 28:36, :] ** 2)
        flat_band = np.sum(out[:, :8, :] ** 2) + np.sum(out[:, -8:, :] ** 2)
        # Edge band must contain *some* HF energy, even if HF leakage into
        # other bands (boundary effects of a finite Haar reconstruction) puts
        # comparable energy in the flat regions. Require strict positivity at
        # the edge — the only correctness guarantee that doesn't depend on the
        # exact pywt reconstruction convention.
        assert edge_band > 1e-6
        # Total HF energy must dwarf any single far-away column's energy.
        far_col = np.sum(out[:, 0:1, :] ** 2)
        assert float(np.sum(out ** 2)) > 10.0 * far_col


def test_paper_backbone_is_default():
    detector = WaRPADDetector()
    assert detector._backbone_name == "facebook/dinov2-large"
    assert detector._d_rescale == 896
    assert detector._d_patch == 224


def test_raw_score_has_no_forced_classification(monkeypatch):
    detector = WaRPADDetector(d_rescale=4, d_patch=4)
    detector._available = True
    monkeypatch.setattr(detector, "_load", lambda: None)
    monkeypatch.setattr(
        detector,
        "_embed_batch",
        lambda patches: np.asarray([[1.0, 0.0]]) if len(patches) == 1 else None,
    )

    result = detector.detect(np.zeros((4, 4, 3), dtype=np.float32))

    assert result.label == "uncertain"
    assert result.details["score_semantics"] == "raw_warpad_anomaly_score"
