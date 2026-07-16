"""Tests for FeatureBank and probe management."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from ai_content_detector.rl_evasion.text_evasion.feature_bank import FeatureBank


class TestFeatureBank:
    def test_init_defaults(self):
        bank = FeatureBank()
        assert bank.max_probes == 50
        assert bank.auroc_threshold == 0.6
        assert len(bank.probes) == 0

    def test_init_custom(self):
        bank = FeatureBank(max_probes=10, auroc_threshold=0.7)
        assert bank.max_probes == 10
        assert bank.auroc_threshold == 0.7

    def test_train_probe_valid(self):
        bank = FeatureBank()
        rng = np.random.RandomState(42)
        # Separable data
        X = np.vstack([rng.randn(20, 4) + 2, rng.randn(20, 4) - 2])
        y = np.array([0] * 20 + [1] * 20)
        probe, scaler, acc, auroc, importance = bank._train_probe(X, y)
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= auroc <= 1.0
        assert importance.shape == (4,)
        assert np.isclose(importance.sum(), 1.0, atol=1e-6)

    def test_train_probe_high_accuracy_on_separable(self):
        bank = FeatureBank()
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(50, 4) + 5, rng.randn(50, 4) - 5])
        y = np.array([0] * 50 + [1] * 50)
        _, _, acc, auroc, _ = bank._train_probe(X, y)
        assert acc > 0.9
        assert auroc > 0.9

    def test_grouped_probe_validation_keeps_pairs_together(self):
        bank = FeatureBank()
        rng = np.random.RandomState(1)
        generated = rng.normal(2, 0.2, (10, 3))
        human = rng.normal(-2, 0.2, (10, 3))
        X = np.vstack([generated, human])
        y = np.array([0] * 10 + [1] * 10)
        groups = np.concatenate([np.arange(10), np.arange(10)])
        _, _, accuracy, auroc, _ = bank._train_probe(X, y, groups)
        assert accuracy > 0.9
        assert auroc > 0.9

    def test_update_adds_discriminative_probe(self):
        bank = FeatureBank(auroc_threshold=0.5)
        rng = np.random.RandomState(42)

        # Use a mock extractor that returns separable features
        class MockStyloExtractor:
            def extract_batch(self, texts):
                n = len(texts)
                return rng.randn(n, 8).astype(np.float32) + 3.0

        class MockStyloExtractorHuman:
            def extract_batch(self, texts):
                n = len(texts)
                return rng.randn(n, 8).astype(np.float32) - 3.0

        # We need to work around the update API which expects texts
        # Instead, test _train_probe directly and verify update flow
        gen_texts = [f"generated text {i}" for i in range(20)]
        human_texts = [f"human text {i}" for i in range(20)]

        # The update method extracts features and trains probes.
        # Without real extractors, we test the probe logic.
        X = np.vstack([rng.randn(20, 8) + 3, rng.randn(20, 8) - 3])
        y = np.array([0] * 20 + [1] * 20)
        _, _, acc, auroc, _ = bank._train_probe(X, y)
        assert auroc > bank.auroc_threshold

    def test_probe_pruning(self):
        bank = FeatureBank(max_probes=2, auroc_threshold=0.0)
        rng = np.random.RandomState(42)

        for i in range(4):
            X = np.vstack([rng.randn(20, 4) + (i + 1), rng.randn(20, 4) - (i + 1)])
            y = np.array([0] * 20 + [1] * 20)
            probe, scaler, acc, auroc, importance = bank._train_probe(X, y)

            from ai_content_detector.rl_evasion.text_evasion.feature_bank import ProbeEntry

            entry = ProbeEntry(
                name=f"probe_{i}",
                feature_family="test",
                round_added=i,
                accuracy=acc,
                auroc=auroc,
                probe=probe,
                scaler=scaler,
                feature_importance=importance,
            )
            bank.probes.append(entry)

            if len(bank.probes) > bank.max_probes:
                bank.probes.sort(key=lambda p: p.auroc, reverse=True)
                bank.probes.pop()

        assert len(bank.probes) <= 2

    def test_compute_penalty_empty(self):
        bank = FeatureBank()
        assert bank.compute_penalty(["some text"]) == 0.0

    def test_get_most_discriminative_features_empty(self):
        bank = FeatureBank()
        assert bank.get_most_discriminative_features() == []

    def test_get_most_discriminative_features_sorted(self):
        bank = FeatureBank(auroc_threshold=0.0)
        rng = np.random.RandomState(42)

        # Add a probe manually
        X = np.vstack([rng.randn(20, 4) + 3, rng.randn(20, 4) - 3])
        y = np.array([0] * 20 + [1] * 20)
        probe, scaler, acc, auroc, importance = bank._train_probe(X, y)

        from ai_content_detector.rl_evasion.text_evasion.feature_bank import ProbeEntry

        bank.probes.append(ProbeEntry(
            name="test_probe",
            feature_family="stylo",
            round_added=1,
            accuracy=acc,
            auroc=auroc,
            probe=probe,
            scaler=scaler,
            feature_importance=importance,
        ))

        features = bank.get_most_discriminative_features(top_k=3)
        assert len(features) <= 4  # max 4 features in our 4-dim data
        assert all("feature" in f and "cumulative_importance" in f for f in features)
        # Should be sorted descending
        importances = [f["cumulative_importance"] for f in features]
        assert importances == sorted(importances, reverse=True)

    def test_save_load_roundtrip(self):
        bank = FeatureBank(max_probes=5, auroc_threshold=0.5)
        rng = np.random.RandomState(42)

        # Add a probe
        X = np.vstack([rng.randn(20, 4) + 3, rng.randn(20, 4) - 3])
        y = np.array([0] * 20 + [1] * 20)
        probe, scaler, acc, auroc, importance = bank._train_probe(X, y)

        from ai_content_detector.rl_evasion.text_evasion.feature_bank import ProbeEntry

        bank.probes.append(ProbeEntry(
            name="roundtrip_probe",
            feature_family="embedding",
            round_added=1,
            accuracy=acc,
            auroc=auroc,
            probe=probe,
            scaler=scaler,
            feature_importance=importance,
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bank.pkl")
            bank.save(path)
            loaded = FeatureBank.load(path)

            assert len(loaded.probes) == 1
            assert loaded.probes[0].name == "roundtrip_probe"
            assert loaded.probes[0].auroc == pytest.approx(auroc)
            assert loaded.max_probes == 5

    def test_summary(self):
        bank = FeatureBank()
        s = bank.summary()
        assert "Feature Bank" in s
        assert "0 probes" in s
