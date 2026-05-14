"""Persistent Feature Bank for cumulative robustness.

After each MultiSPIN training round, train linear probe classifiers on
feature vectors to identify which features the current model gives itself
away on. Accumulate these probes so future rounds are penalized against
being separable by any historically discriminative feature.

Inspired by Elastic Weight Consolidation (Kirkpatrick et al., 2017)
applied to feature-matching objectives rather than to weights.

Reference: todo/rl_evasion_research_directions_v2.md — "The persistent feature bank"
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ProbeEntry:
    """A stored probe classifier and its metadata."""

    name: str
    feature_family: str  # "stylometric", "embedding", "surprisal"
    round_added: int
    accuracy: float
    auroc: float
    probe: LogisticRegression = field(repr=False)
    scaler: StandardScaler = field(repr=False)
    feature_importance: np.ndarray = field(repr=False)


class FeatureBank:
    """Persistent bank of discriminative probe classifiers.

    Workflow per round:
    1. Generate outputs from current model.
    2. Extract feature vectors (stylometric, embedding, surprisal).
    3. Train linear probes: human vs. current-model on each feature family.
    4. If probe is discriminative (AUROC > threshold), add to bank.
    5. Next round's training loss includes penalty from all banked probes.
    """

    def __init__(
        self,
        max_probes: int = 50,
        auroc_threshold: float = 0.6,
        save_dir: Optional[str] = None,
    ):
        self.max_probes = max_probes
        self.auroc_threshold = auroc_threshold
        self.save_dir = save_dir
        self.probes: List[ProbeEntry] = []
        self._round = 0

    def update(
        self,
        generated_texts: List[str],
        human_texts: List[str],
        stylo_extractor=None,
        emb_extractor=None,
    ) -> Dict[str, float]:
        """Train probes on current round's data and add discriminative ones.

        Returns metrics about which features are still discriminative.
        """
        self._round += 1
        metrics = {}

        # Prepare labels
        n_gen = len(generated_texts)
        n_hum = min(len(human_texts), n_gen)
        labels = np.array([0] * n_gen + [1] * n_hum)  # 0=generated, 1=human

        # Extract features for each family
        families = {}

        if stylo_extractor is not None:
            gen_stylo = stylo_extractor.extract_batch(generated_texts)
            hum_stylo = stylo_extractor.extract_batch(human_texts[:n_hum])
            families["stylometric"] = np.vstack([gen_stylo, hum_stylo])

        if emb_extractor is not None:
            gen_emb = emb_extractor.extract_batch(generated_texts)
            hum_emb = emb_extractor.extract_batch(human_texts[:n_hum])
            families["embedding"] = np.vstack([gen_emb, hum_emb])

        # Train a probe per family
        for family_name, X in families.items():
            probe, scaler, acc, auroc, importance = self._train_probe(X, labels)

            metrics[f"{family_name}_probe_accuracy"] = acc
            metrics[f"{family_name}_probe_auroc"] = auroc

            if auroc >= self.auroc_threshold:
                entry = ProbeEntry(
                    name=f"{family_name}_round{self._round}",
                    feature_family=family_name,
                    round_added=self._round,
                    accuracy=acc,
                    auroc=auroc,
                    probe=probe,
                    scaler=scaler,
                    feature_importance=importance,
                )
                self.probes.append(entry)
                logger.info(
                    "Added probe: %s (AUROC=%.3f, acc=%.3f)",
                    entry.name, auroc, acc,
                )

                # Prune if over capacity (remove oldest with lowest AUROC)
                if len(self.probes) > self.max_probes:
                    self.probes.sort(key=lambda p: p.auroc, reverse=True)
                    removed = self.probes.pop()
                    logger.info("Pruned probe: %s (AUROC=%.3f)", removed.name, removed.auroc)
            else:
                logger.info(
                    "Probe %s AUROC=%.3f below threshold=%.3f — not added.",
                    family_name, auroc, self.auroc_threshold,
                )

        if self.save_dir:
            self.save()

        return metrics

    def _train_probe(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[LogisticRegression, StandardScaler, float, float, np.ndarray]:
        """Train a linear probe classifier and return metrics."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        probe = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        probe.fit(X_scaled, y)

        y_pred = probe.predict(X_scaled)
        y_prob = probe.predict_proba(X_scaled)[:, 1]

        acc = accuracy_score(y, y_pred)
        try:
            auroc = roc_auc_score(y, y_prob)
        except ValueError:
            auroc = 0.5

        # Feature importance: absolute coefficients
        importance = np.abs(probe.coef_).flatten()
        importance = importance / (importance.sum() + 1e-8)

        return probe, scaler, acc, auroc, importance

    def compute_penalty(
        self,
        generated_texts: List[str],
        stylo_extractor=None,
        emb_extractor=None,
    ) -> float:
        """Compute penalty from all banked probes.

        Returns a scalar penalty: how easily the banked probes can still
        distinguish the current model's output from human text.
        Higher penalty = more detectable.
        """
        if not self.probes:
            return 0.0

        penalties = []

        for probe_entry in self.probes:
            family = probe_entry.feature_family

            if family == "stylometric" and stylo_extractor is not None:
                X = stylo_extractor.extract_batch(generated_texts)
            elif family == "embedding" and emb_extractor is not None:
                X = emb_extractor.extract_batch(generated_texts)
            else:
                continue

            X_scaled = probe_entry.scaler.transform(X)
            # Probability of being classified as "generated" (class 0)
            probs_generated = probe_entry.probe.predict_proba(X_scaled)[:, 0]
            # Penalty: how confidently the probe identifies text as generated
            penalty = float(np.mean(probs_generated))
            penalties.append(penalty)

        return float(np.mean(penalties)) if penalties else 0.0

    def get_most_discriminative_features(self, top_k: int = 10) -> List[Dict]:
        """Return the top-k most discriminative features across all probes."""
        if not self.probes:
            return []

        feature_scores = {}
        for entry in self.probes:
            for i, imp in enumerate(entry.feature_importance):
                key = f"{entry.feature_family}_feat{i}"
                if key not in feature_scores:
                    feature_scores[key] = 0.0
                feature_scores[key] += imp * entry.auroc  # weight by probe quality

        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"feature": k, "cumulative_importance": v} for k, v in sorted_features[:top_k]]

    def save(self, path: Optional[str] = None):
        """Save the feature bank to disk."""
        path = path or os.path.join(self.save_dir, "feature_bank.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Feature bank saved to %s (%d probes)", path, len(self.probes))

    @classmethod
    def load(cls, path: str) -> "FeatureBank":
        """Load a feature bank from disk."""
        with open(path, "rb") as f:
            bank = pickle.load(f)
        logger.info("Feature bank loaded from %s (%d probes)", path, len(bank.probes))
        return bank

    def summary(self) -> str:
        """Return a human-readable summary of the feature bank."""
        lines = [f"Feature Bank: {len(self.probes)} probes, round {self._round}"]
        for entry in self.probes:
            lines.append(
                f"  {entry.name}: AUROC={entry.auroc:.3f}, acc={entry.accuracy:.3f}, "
                f"top features: {np.argsort(entry.feature_importance)[::-1][:3].tolist()}"
            )
        return "\n".join(lines)
