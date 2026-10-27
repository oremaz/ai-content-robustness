"""Window smoothing around document-level text detectors."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from .ensemble import BaseDetector, DetectionResult

logger = logging.getLogger("ai_content_detector.detectors.text_detectors")


class WindowSmoothedTextDetector(BaseDetector):
    """Fixed local-window neighbor smoother for detector scores.

    Wu et al. (ICLR 2026) show that token-level metric-detector scores benefit
    from two structural priors: nearby scores should be similar, while initial
    positions are less stable. Their paper implements this with a Markov random
    field and mean-field approximation.

    Most detectors in this package expose document scores, not raw token scores.
    This wrapper therefore applies the same idea to overlapping local windows:
    score each window with a base detector or callable, run a mean-field-style
    neighbor smoother, downweight unstable early windows, and aggregate the
    calibrated local probabilities. Use it as a lightweight calibration layer,
    not as a replacement for a paper-exact token-level MRF implementation.
    """

    __module__ = "ai_content_detector.detectors.text_detectors"

    name = "Window-Smoothed Detector"
    modality = "text"

    def __init__(
        self,
        base_detector: Optional[BaseDetector] = None,
        score_fn: Optional[Callable[[str], float]] = None,
        window_size: int = 64,
        stride: int = 32,
        neighbor_weight: float = 0.8,
        unary_weight: float = 1.0,
        initial_discount: float = 0.5,
        iterations: int = 4,
    ):
        if base_detector is None and score_fn is None:
            raise ValueError("WindowSmoothedTextDetector needs base_detector or score_fn")
        self.base_detector = base_detector
        self.score_fn = score_fn
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.neighbor_weight = float(neighbor_weight)
        self.unary_weight = float(unary_weight)
        self.initial_discount = float(initial_discount)
        self.iterations = int(iterations)
        if base_detector is not None:
            self.name = f"Window-Smoothed {base_detector.name}"

    def is_available(self) -> bool:
        if self.base_detector is not None:
            return self.base_detector.is_available()
        return self.score_fn is not None

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def calibrate_scores(
        raw_scores: Sequence[float],
        neighbor_weight: float = 0.8,
        unary_weight: float = 1.0,
        initial_discount: float = 0.5,
        iterations: int = 4,
    ) -> np.ndarray:
        """Mean-field-style calibration for a 1D score chain."""
        raw = np.asarray(raw_scores, dtype=np.float64)
        if raw.size == 0:
            return raw.astype(np.float32)
        if raw.size == 1:
            return np.clip(raw, 0.0, 1.0).astype(np.float32)

        unary = WindowSmoothedTextDetector._logit(raw)
        weights = np.ones_like(raw)
        weights[0] = float(initial_discount)
        if raw.size > 2:
            weights[1] = max(float(initial_discount), 0.75)
        q = np.clip(raw, 1e-5, 1.0 - 1e-5)

        for _ in range(max(1, int(iterations))):
            prev = np.concatenate(([q[0]], q[:-1]))
            nxt = np.concatenate((q[1:], [q[-1]]))
            # Binary pairwise MRF mean-field update: neighbor fields push each
            # node toward adjacent beliefs, while the unary term preserves the
            # detector's own evidence.
            neighbor_field = (prev - 0.5) + (nxt - 0.5)
            logits = float(unary_weight) * weights * unary + float(neighbor_weight) * neighbor_field
            q = WindowSmoothedTextDetector._sigmoid(logits)
        return np.clip(q, 0.0, 1.0).astype(np.float32)

    def _windows(self, text: str) -> List[str]:
        words = text.split()
        if len(words) <= self.window_size:
            return [text]
        stride = max(1, self.stride)
        windows = []
        for start in range(0, len(words), stride):
            chunk = words[start : start + self.window_size]
            if len(chunk) < max(8, self.window_size // 4) and windows:
                break
            windows.append(" ".join(chunk))
            if start + self.window_size >= len(words):
                break
        return windows or [text]

    def _score_window(self, window: str) -> float:
        if self.score_fn is not None:
            return float(self.score_fn(window))
        if self.base_detector is None:
            raise RuntimeError("No base detector or score_fn configured")
        result = self.base_detector.detect(window)
        if isinstance(result, list):
            result = result[0]
        return float(result.score)

    def detect(self, content: Union[str, List[str]]) -> Union[DetectionResult, List[DetectionResult]]:
        if not self.is_available():
            raise RuntimeError("Markov calibration base detector is unavailable")

        is_single = isinstance(content, str)
        texts = [content] if is_single else content

        results = []
        for text in texts:
            windows = self._windows(text)
            raw_scores = [max(0.0, min(1.0, self._score_window(w))) for w in windows]
            calibrated = self.calibrate_scores(
                raw_scores,
                neighbor_weight=self.neighbor_weight,
                unary_weight=self.unary_weight,
                initial_discount=self.initial_discount,
                iterations=self.iterations,
            )

            if len(calibrated) > 1:
                weights = np.ones(len(calibrated), dtype=np.float32)
                weights[0] = float(self.initial_discount)
                if len(weights) > 2:
                    weights[1] = max(float(self.initial_discount), 0.75)
                ai_score = float(np.average(calibrated, weights=weights))
            else:
                ai_score = float(calibrated[0])
            ai_score = self._complexity_penalty(text, ai_score)
            results.append(DetectionResult(
                score=ai_score,
                label=DetectionResult.label_from_score(ai_score),
                detector_name=self.name,
                details={
                    "raw_window_scores": [float(s) for s in raw_scores],
                    "calibrated_window_scores": [float(s) for s in calibrated],
                    "num_windows": len(windows),
                    "window_size": self.window_size,
                    "stride": self.stride,
                    "neighbor_weight": self.neighbor_weight,
                    "initial_discount": self.initial_discount,
                },
            ))

        return results[0] if is_single else results


MarkovCalibratedTextDetector = WindowSmoothedTextDetector
