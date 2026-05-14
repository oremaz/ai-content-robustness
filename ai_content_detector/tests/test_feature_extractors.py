"""Tests for StylometricExtractor and compute_mmd."""

from __future__ import annotations

import numpy as np
import pytest

from ai_content_detector.rl_evasion.text_evasion.multispin import (
    StylometricExtractor,
    compute_mmd,
)

spacy = pytest.importorskip("spacy", reason="spacy not installed")


class TestStylometricExtractor:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.extractor = StylometricExtractor()

    def test_extract_returns_8_features(self, human_texts):
        vec = self.extractor.extract(human_texts[0])
        assert vec.shape == (8,)
        assert vec.dtype == np.float32

    def test_extract_batch(self, human_texts):
        batch = self.extractor.extract_batch(human_texts[:3])
        assert batch.shape == (3, 8)

    def test_ttr_in_range(self, human_texts):
        vec = self.extractor.extract(human_texts[0])
        ttr = vec[1]  # index 1 = TTR
        assert 0.0 <= ttr <= 1.0

    def test_burstiness_nonnegative(self, human_texts):
        vec = self.extractor.extract(human_texts[0])
        burstiness = vec[0]
        assert burstiness >= 0.0

    def test_punct_density_in_range(self, human_texts):
        vec = self.extractor.extract(human_texts[0])
        punct_density = vec[6]  # index 6
        assert 0.0 <= punct_density <= 1.0

    def test_hapax_ratio_in_range(self, human_texts):
        vec = self.extractor.extract(human_texts[0])
        hapax_ratio = vec[7]  # index 7
        assert 0.0 <= hapax_ratio <= 1.0

    def test_fw_ratio_in_range(self, human_texts):
        vec = self.extractor.extract(human_texts[0])
        fw_ratio = vec[4]  # index 4
        assert 0.0 <= fw_ratio <= 1.0

    def test_short_text(self):
        vec = self.extractor.extract("Hello world.")
        assert vec.shape == (8,)
        assert np.all(np.isfinite(vec))

    def test_single_word(self):
        vec = self.extractor.extract("Hello")
        assert vec.shape == (8,)
        assert np.all(np.isfinite(vec))

    def test_different_texts_different_features(self, human_texts, ai_texts):
        h_vec = self.extractor.extract(human_texts[0])
        a_vec = self.extractor.extract(ai_texts[0])
        # Not identical
        assert not np.allclose(h_vec, a_vec)


class TestComputeMMD:
    def test_identical_distributions_near_zero(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 4)
        mmd = compute_mmd(X, X.copy(), gamma=1.0)
        assert mmd == pytest.approx(0.0, abs=0.05)

    def test_different_distributions_positive(self, separable_features):
        X, Y = separable_features
        mmd = compute_mmd(X, Y, gamma=0.1)
        assert mmd > 0.0

    def test_separable_distributions_large_mmd(self, separable_features):
        X, Y = separable_features
        mmd = compute_mmd(X, Y, gamma=0.1)
        assert mmd > 0.1

    def test_empty_returns_zero(self):
        X = np.array([]).reshape(0, 4)
        Y = np.random.randn(10, 4)
        assert compute_mmd(X, Y) == 0.0
        assert compute_mmd(Y, X) == 0.0

    def test_both_empty_returns_zero(self):
        X = np.array([]).reshape(0, 4)
        assert compute_mmd(X, X) == 0.0

    def test_nonnegative(self, separable_features):
        X, Y = separable_features
        assert compute_mmd(X, Y) >= 0.0

    def test_symmetric(self, separable_features):
        X, Y = separable_features
        assert compute_mmd(X, Y) == pytest.approx(compute_mmd(Y, X), abs=1e-6)

    def test_gamma_affects_result(self, separable_features):
        X, Y = separable_features
        mmd_low = compute_mmd(X, Y, gamma=0.01)
        mmd_high = compute_mmd(X, Y, gamma=10.0)
        # Different gammas should give different results
        assert mmd_low != pytest.approx(mmd_high, abs=1e-3)
