"""Reward functions for text evasion RL training.

Each reward function takes generated text and returns a scalar reward.
The composite reward combines evasion, semantic preservation, and quality.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DetectorReward:
    """Reward based on evasion of a text detector.

    reward = 1 - detector_score (higher is better = more evasion).

    A failing detector raises by default — silent fall-back to 0.5 masks broken
    runs and makes training results unreliable. Pass ``silent_fallback=True``
    only for throwaway debugging.
    """

    def __init__(
        self,
        detector_fn: Callable[[str], float],
        name: str = "detector",
        silent_fallback: bool = False,
    ):
        self.detector_fn = detector_fn
        self.name = name
        self.silent_fallback = silent_fallback

    def __call__(self, text: str) -> float:
        try:
            score = float(self.detector_fn(text))
        except Exception as e:
            logger.error("Detector %s failed: %s", self.name, e)
            if self.silent_fallback:
                return 0.5
            raise
        return 1.0 - score


class SemanticSimilarityReward:
    """Reward for preserving semantic similarity with the source text.

    Uses a frozen sentence encoder (E5 or SBERT) for cosine similarity.
    """

    def __init__(self, model_name: str = "intfloat/e5-base-v2", device: str = "auto"):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(device)
        self._model.eval()

    def _embed(self, text: str) -> np.ndarray:
        # E5 model card prescribes the "query: " / "passage: " instruction prefix
        # and mean-pooling over non-pad tokens. Using CLS alone underperforms.
        prefixed = f"query: {text}" if "e5" in self._model_name.lower() else text
        tokens = self._tokenizer(
            prefixed, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model(**tokens)
            hidden = outputs.last_hidden_state  # (1, L, D)
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = (summed / counts).cpu().numpy()
        return pooled / (np.linalg.norm(pooled) + 1e-8)

    def __call__(self, source: str, generated: str) -> float:
        self._load()
        emb_src = self._embed(source)
        emb_gen = self._embed(generated)
        return float(np.dot(emb_src.flatten(), emb_gen.flatten()))


class CompositeReward:
    """Combines multiple reward signals into a single scalar.

    reward = w_evasion * mean(evasion_rewards) +
             w_semantic * semantic_sim +
             w_quality * quality_reward
    """

    def __init__(
        self,
        detector_rewards: List[DetectorReward],
        semantic_reward: Optional[SemanticSimilarityReward] = None,
        weights: Optional[Dict[str, float]] = None,
        min_output_tokens: int = 20,
    ):
        self.detector_rewards = detector_rewards
        if not detector_rewards:
            raise ValueError("CompositeReward requires at least one detector reward")
        self.semantic_reward = semantic_reward or SemanticSimilarityReward()
        self.weights = weights or {
            "evasion": 0.6,
            "semantic": 0.3,
            "quality": 0.1,
        }
        self.min_output_tokens = int(min_output_tokens)

    def __call__(self, source: str, generated: str) -> Dict[str, float]:
        # Reject degenerate (too-short) outputs so the RL signal punishes them
        # rather than letting them slip through via zero-quality + high evasion.
        gen_tokens = generated.split()
        if len(gen_tokens) < self.min_output_tokens:
            return {
                "total": 0.0,
                "evasion": 0.0,
                "semantic": 0.0,
                "quality": 0.0,
                "rejected": True,
                "per_detector": {dr.name: 0.0 for dr in self.detector_rewards},
            }

        # Evasion rewards
        evasion_scores = [dr(generated) for dr in self.detector_rewards]
        mean_evasion = float(np.mean(evasion_scores)) if evasion_scores else 0.5

        # Semantic similarity
        semantic_sim = self.semantic_reward(source, generated)

        # Quality bonus: length ratio penalty (avoid degenerate outputs).
        # Quadratic, so extreme-length outputs are strongly penalized even after
        # the min-length gate above.
        src_len = max(len(source.split()), 1)
        len_ratio = len(gen_tokens) / src_len
        quality = max(0.0, 1.0 - (1.0 - len_ratio) ** 2)

        total = (
            self.weights["evasion"] * mean_evasion
            + self.weights["semantic"] * semantic_sim
            + self.weights["quality"] * quality
        )

        return {
            "total": float(total),
            "evasion": float(mean_evasion),
            "semantic": float(semantic_sim),
            "quality": float(quality),
            "rejected": False,
            "per_detector": {
                dr.name: float(score) for dr, score in zip(self.detector_rewards, evasion_scores)
            },
        }


def build_detector_reward_from_name(name: str, device: str = "auto") -> DetectorReward:
    """Factory: create a DetectorReward from a detector name."""

    if name == "binoculars":
        from ai_content_detector.detectors.text_detectors import BinocularsDetector
        det = BinocularsDetector(device=device)
        return DetectorReward(lambda text, d=det: d.detect(text).score, name="binoculars")

    elif name == "fast_detect_gpt":
        from ai_content_detector.detectors.text_detectors import FastDetectGPTDetector
        det = FastDetectGPTDetector(device=device)
        return DetectorReward(lambda text, d=det: d.detect(text).score, name="fast_detect_gpt")

    elif name == "diveye":
        from ai_content_detector.detectors.text_detectors import DivEyeDetector
        det = DivEyeDetector(device=device)
        return DetectorReward(lambda text, d=det: d.detect(text).score, name="diveye")

    elif name in ("disrupt_recover", "dr"):
        from ai_content_detector.detectors.text_detectors import DisruptRecoverDetector
        det = DisruptRecoverDetector()
        return DetectorReward(lambda text, d=det: d.detect(text).score, name="disrupt_recover")

    elif name == "roberta_classifier":
        # Use a pretrained RoBERTa-based detector.
        return _build_roberta_detector_reward(device)

    elif name == "style_embedding":
        from ai_content_detector.detectors.style_detector import StyleEmbeddingDetector
        det = StyleEmbeddingDetector(device=device)
        return DetectorReward(lambda text, d=det: d.detect(text).score, name="style_embedding")

    else:
        raise ValueError(f"Unknown detector name: {name}")


def _build_roberta_detector_reward(device: str) -> DetectorReward:
    """Build a RoBERTa-based detector reward.

    Uses a configured RoBERTa classifier if available,
    otherwise falls back to a HuggingFace pipeline.
    """
    try:
        from transformers import pipeline

        pipe = pipeline(
            "text-classification",
            model="roberta-base-openai-detector",
            device=0 if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else -1,
        )

        def detect_fn(text: str) -> float:
            result = pipe(text, truncation=True, max_length=512)
            # Higher score for "Fake" label = more AI-like
            for r in result:
                if r["label"].lower() in ("fake", "machine", "ai", "generated"):
                    return r["score"]
            return 1.0 - result[0]["score"]

        return DetectorReward(detect_fn, name="roberta_classifier")

    except Exception as e:
        # Fail loudly: a silent dummy reward turns a broken detector into silent
        # noise that will train the model to do nothing useful.
        raise RuntimeError(
            f"Could not load RoBERTa detector (roberta-base-openai-detector): {e}. "
            "Either install/download it, or drop 'roberta_classifier' from "
            "TextEvasionConfig.detector_names."
        ) from e
