"""Base detector interface and ensemble aggregation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Sequence

import numpy as np


@dataclass
class DetectionResult:
    """Result returned by every detector."""

    score: float  # 0.0 = certainly human, 1.0 = certainly AI
    label: str  # "human", "ai", or "uncertain"
    detector_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def label_from_score(score: float, ai_threshold: float = 0.65, human_threshold: float = 0.35) -> str:
        if score >= ai_threshold:
            return "ai"
        if score <= human_threshold:
            return "human"
        return "uncertain"


class BaseDetector(ABC):
    """Common interface for all text and image detectors."""

    name: str = "base"
    modality: str = "text"  # "text" or "image"

    @abstractmethod
    def detect(self, content: Any) -> Union[DetectionResult, List[DetectionResult]]:
        ...

    def is_available(self) -> bool:
        """Return True if this detector can run (deps installed, checkpoint exists, etc.)."""
        return True

    # Instance-attribute names that may hold heavy loaded resources
    # (models, tokenizers, pipelines). unload() clears any that are present.
    _RESOURCE_ATTRS = (
        "_model", "_tokenizer", "_vectorizer", "_predictor", "_processor",
        "_pipe", "_pipeline", "_observer", "_performer", "_reference_model",
        "_clip_model", "_clip_processor", "_uncond_embed",
    )

    def unload(self) -> None:
        """Release loaded models / heavy resources so GPU + RAM can be reclaimed.

        Safe to call when nothing is loaded. After unloading, the detector
        returns to its lazy state and reloads on the next detect() call.
        Prevents GPU memory from accumulating across repeated analyses.
        """
        freed = False
        for attr in self._RESOURCE_ATTRS:
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)
                freed = True
        # Reset a cached "available" flag (only if it was True) so the
        # model reloads on next use; leave False/None untouched.
        if getattr(self, "_available", None):
            self._available = None
        if not freed:
            return
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


    def _dynamic_threshold(self, length: int, base_threshold: float) -> float:
        """Dynamic thresholding based on input length to mitigate short-text bias."""
        if length < 50:
            return base_threshold * 1.2  # Stricter for very short text
        elif length < 150:
            return base_threshold * 1.1
        elif length > 500:
            return base_threshold * 0.95 # Lenient for long text
        return base_threshold
        
    def _complexity_penalty(self, text: str, ai_score: float) -> float:
        """Apply a confidence penalty if text exhibits extremely low lexical complexity (ESL bias)."""
        words = text.split()
        if len(words) < 10:
            return ai_score
            
        ttr = len(set(words)) / len(words)  # Type-Token Ratio
        try:
            import textstat
            flesch = textstat.flesch_kincaid_grade(text)
        except ImportError:
            flesch = 10.0
            
        # If text is extremely simple (low grade level, high repetition), reduce AI confidence
        if ttr < 0.4 and flesch < 6.0:
            return ai_score * 0.8  # 20% penalty
        return ai_score


class EnsembleAggregator:
    """Aggregate scores from multiple detectors."""

    def __init__(
        self,
        detectors: List[BaseDetector],
        weights: Optional[Dict[str, float]] = None,
        method: str = "weighted_average",
    ):
        self.detectors = [d for d in detectors if d.is_available()]
        if not self.detectors:
            raise ValueError("EnsembleAggregator requires at least one available detector")
        self.weights = weights or {d.name: 1.0 for d in self.detectors}
        self.method = method
        self._meta_classifier = None

    def optimize_weights(self, val_texts: List[Any], val_labels: List[int]) -> None:
        """Data-driven ensemble optimization (OUTFOX framework inspired).
        
        Learns optimal weights using Logistic Regression on a calibration dataset.
        """
        from sklearn.linear_model import LogisticRegression
        if len(val_texts) != len(val_labels) or not val_texts:
            raise ValueError("Calibration texts and labels must be nonempty and aligned")
        if len(set(val_labels)) < 2:
            raise ValueError("Ensemble calibration requires both human and AI labels")
        X = []
        for text in val_texts:
            scores = []
            for det in self.detectors:
                try:
                    res = det.detect(text)
                    score = res[0].score if isinstance(res, list) else res.score
                except Exception as error:
                    raise RuntimeError(
                        f"Detector {det.name} failed during ensemble calibration"
                    ) from error
                scores.append(score)
            X.append(scores)
                
        self._meta_classifier = LogisticRegression(class_weight="balanced", random_state=42)
        self._meta_classifier.fit(X, val_labels)
        coeffs = self._meta_classifier.coef_[0]
        self.weights = {det.name: float(w) for det, w in zip(self.detectors, coeffs)}
        self.method = "learned_weights"

    def detect(self, content: Any) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        # Handle batch inputs
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], str):
            return [self._detect_single(c) for c in content]
        return self._detect_single(content)

    def _detect_single(self, content: Any) -> Dict[str, Any]:
        results: List[DetectionResult] = []
        for det in self.detectors:
            try:
                r = det.detect(content)
                if isinstance(r, list):
                    r = r[0]  # Just take first if accidentally returned list for single item
                r.detector_name = det.name
                results.append(r)
            except Exception as e:
                results.append(
                    DetectionResult(
                        score=0.5,
                        label="error",
                        detector_name=det.name,
                        details={"error": str(e)},
                    )
                )

        aggregate_score = self._aggregate(results)
        valid = [result for result in results if result.label != "error"]
        if self.method == "learned_weights":
            aggregate_label = DetectionResult.label_from_score(aggregate_score, 0.5, 0.5)
            semantics = "calibrated_probability"
        else:
            ai_votes = sum(result.label == "ai" for result in valid)
            human_votes = sum(result.label == "human" for result in valid)
            aggregate_label = "ai" if ai_votes > human_votes else "human" if human_votes > ai_votes else "uncertain"
            semantics = "uncalibrated_ensemble_score"

        return {
            "aggregate_score": aggregate_score,
            "aggregate_label": aggregate_label,
            "aggregate_score_semantics": semantics,
            "per_detector": results,
        }

    def _aggregate(self, results: List[DetectionResult]) -> float:
        valid = [r for r in results if r.label != "error"]
        if not valid:
            raise RuntimeError("Every detector failed; no aggregate score is available")

        if self.method == "majority_vote":
            votes = [1.0 if r.label == "ai" else 0.0 for r in valid]
            return float(np.mean(votes))
            
        if self.method == "learned_weights" and self._meta_classifier is not None:
            # Reconstruct feature vector aligned with detectors
            scores = [0.5] * len(self.detectors)
            det_idx = {d.name: i for i, d in enumerate(self.detectors)}
            for r in valid:
                if r.detector_name in det_idx:
                    scores[det_idx[r.detector_name]] = r.score
            return float(self._meta_classifier.predict_proba([scores])[0][1])

        # weighted_average (default)
        scores, ws = [], []
        for r in valid:
            scores.append(r.score)
            ws.append(self.weights.get(r.detector_name, 1.0))
        ws_arr = np.array(ws)
        if ws_arr.sum() > 0:
            ws_arr = ws_arr / ws_arr.sum()
        else:
            ws_arr = np.ones_like(ws_arr) / len(ws_arr)
        return float(np.dot(scores, ws_arr))
