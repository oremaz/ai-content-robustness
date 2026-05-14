"""Shared fixtures for ai_content_detector tests.

All fixtures are lightweight — no GPU, no model downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from ai_content_detector.detectors.ensemble import BaseDetector, DetectionResult


# ---------------------------------------------------------------------------
# Dummy detectors
# ---------------------------------------------------------------------------


class DummyDetector(BaseDetector):
    """Detector that returns a fixed score."""

    def __init__(self, score: float = 0.8, name: str = "dummy"):
        self._score = score
        self.name = name
        self.modality = "text"

    def detect(self, content: Any) -> DetectionResult:
        return DetectionResult(
            score=self._score,
            label=DetectionResult.label_from_score(self._score),
            detector_name=self.name,
        )


class FailingDetector(BaseDetector):
    """Detector that always raises."""

    name = "failing"
    modality = "text"

    def detect(self, content: Any) -> DetectionResult:
        raise RuntimeError("detector crashed")


class UnavailableDetector(BaseDetector):
    """Detector that reports itself as unavailable."""

    name = "unavailable"
    modality = "text"

    def detect(self, content: Any) -> DetectionResult:
        return DetectionResult(score=0.5, label="uncertain")

    def is_available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------


@pytest.fixture
def human_texts() -> list[str]:
    return [
        "The morning sun cast long shadows across the kitchen floor as Maria poured "
        "her second cup of coffee. She glanced at the clock — already eight-fifteen. "
        "The bus wouldn't wait, and neither would her boss.",
        "I went to the store yesterday and bought some apples. They were on sale, "
        "two pounds for a dollar. Not bad, considering the season.",
        "Rain hammered the tin roof all night. By dawn the creek had swollen past "
        "its banks, swallowing the footbridge whole.",
        "My grandmother used to say that a watched pot never boils. I think she "
        "was really talking about patience, not cooking.",
        "The committee voted 7-3 to approve the new zoning ordinance, despite "
        "objections from several neighborhood associations.",
    ]


@pytest.fixture
def ai_texts() -> list[str]:
    return [
        "Artificial intelligence has revolutionized numerous industries, offering "
        "unprecedented capabilities in data analysis, pattern recognition, and "
        "automated decision-making processes.",
        "The implementation of machine learning algorithms requires careful "
        "consideration of various factors, including data quality, model "
        "architecture, and computational resources.",
        "In the realm of natural language processing, transformer-based models "
        "have demonstrated remarkable performance across a wide range of tasks, "
        "from text generation to sentiment analysis.",
        "Furthermore, the integration of deep learning techniques with "
        "traditional statistical methods has yielded significant improvements "
        "in predictive accuracy and model interpretability.",
        "The convergence of cloud computing and artificial intelligence has "
        "enabled organizations to deploy sophisticated analytical solutions "
        "at scale, transforming business operations.",
    ]


@pytest.fixture
def sample_prompts() -> list[str]:
    return [
        "Write about the impact of technology on education.",
        "Describe a typical morning routine.",
        "Explain how climate change affects agriculture.",
    ]


# ---------------------------------------------------------------------------
# Dummy detector fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_high_detector() -> DummyDetector:
    """Detector that always scores 0.9 (AI)."""
    return DummyDetector(score=0.9, name="high")


@pytest.fixture
def dummy_low_detector() -> DummyDetector:
    """Detector that always scores 0.1 (human)."""
    return DummyDetector(score=0.1, name="low")


@pytest.fixture
def dummy_mid_detector() -> DummyDetector:
    """Detector that always scores 0.5 (uncertain)."""
    return DummyDetector(score=0.5, name="mid")


@pytest.fixture
def failing_detector() -> FailingDetector:
    return FailingDetector()


# ---------------------------------------------------------------------------
# Feature arrays
# ---------------------------------------------------------------------------


@pytest.fixture
def random_features() -> np.ndarray:
    """Random feature matrix (20 samples x 8 features)."""
    rng = np.random.RandomState(42)
    return rng.randn(20, 8).astype(np.float32)


@pytest.fixture
def separable_features() -> tuple[np.ndarray, np.ndarray]:
    """Two clearly separable feature distributions."""
    rng = np.random.RandomState(42)
    X = rng.randn(30, 8).astype(np.float32) + 2.0
    Y = rng.randn(30, 8).astype(np.float32) - 2.0
    return X, Y
