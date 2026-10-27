"""Image reward components and prompt-local advantage statistics."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch


def reference_similarity_reward(kl, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Reference KL scale must be finite and positive")
    values = torch.as_tensor(kl)
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Reference KL must be finite and nonnegative")
    return torch.reciprocal(1 + values / scale)


def component_statistics(values, weight):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()),
            "std": float(values.std()), "weighted_std": float(weight * values.std()),
            "weighted_range": float(weight * np.ptp(values))}


def rollout_prompt_indices(count, size, repeats, seed):
    if repeats < 1 or size % repeats or size // repeats > count:
        raise ValueError("Rollout must contain whole prompt groups from the training pool")
    return np.repeat(np.random.default_rng(seed).permutation(count)[:size // repeats], repeats)


def grouped_advantages(prompts, rewards):
    prompts, rewards = np.asarray(prompts), np.asarray(rewards, dtype=float)
    advantages = np.zeros_like(rewards)
    for prompt in np.unique(prompts):
        mask = prompts == prompt
        if mask.sum() < 2:
            raise ValueError("Grouped advantages need at least two images per prompt")
        values = rewards[mask]
        advantages[mask] = (values - values.mean()) / (values.std() + 1e-8)
    return advantages


class PerPromptStatTracker:
    """Track running mean/std of rewards per prompt for advantage normalization.

    advantage = (reward - mean_reward_for_prompt) / (std_reward_for_prompt + eps)
    """

    def __init__(self, buffer_size: int = 32, min_count: int = 2):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self._stats: Dict[str, List[float]] = defaultdict(list)

    def update(self, prompts: List[str], rewards: List[float]) -> np.ndarray:
        """Record rewards and return normalized advantages."""
        advantages = np.zeros(len(rewards))

        rewards_arr = np.asarray(rewards, dtype=np.float64)
        global_std = float(rewards_arr.std())
        global_advantages = (
            (rewards_arr - rewards_arr.mean()) / (global_std + 1e-8)
            if global_std > 1e-8 else np.zeros_like(rewards_arr)
        )
        for i, (prompt, reward) in enumerate(zip(prompts, rewards)):
            self._stats[prompt].append(reward)
            # Keep buffer bounded
            if len(self._stats[prompt]) > self.buffer_size:
                self._stats[prompt] = self._stats[prompt][-self.buffer_size:]

            buf = self._stats[prompt]
            if len(buf) >= self.min_count:
                advantages[i] = (reward - np.mean(buf)) / (np.std(buf) + 1e-8)
            else:
                advantages[i] = global_advantages[i]

        return advantages


class ImageDetectorReward:
    """Compute evasion reward from an image detector.

    reward = 1 - detector_score (higher = less detectable)
    """

    def __init__(self, detector, name: str = "detector"):
        self.detector = detector
        self.name = name

    def __call__(self, image) -> float:
        result = self.detector.detect(image)
        if not 0.0 <= result.score <= 1.0:
            raise ValueError(f"Detector {self.name} returned invalid score {result.score}")
        return 1.0 - result.score


class CompositeImageReward:
    """Combine evasion, CLIP alignment, and aesthetic rewards."""

    def __init__(
        self,
        detector_rewards: List[ImageDetectorReward],
        weights: Optional[Dict[str, float]] = None,
        aesthetic_scorer=None,
        clip_model_id="openai/clip-vit-base-patch32",
    ):
        self.detector_rewards = detector_rewards
        self.weights = weights or {"evasion": 0.5, "clip": 0.3, "aesthetic": 0.2}
        if not detector_rewards:
            raise ValueError("CompositeImageReward requires at least one detector")
        if self.weights.get("aesthetic", 0.0) > 0 and aesthetic_scorer is None:
            raise ValueError("A nonzero aesthetic weight requires an aesthetic_scorer")
        self.aesthetic_scorer = aesthetic_scorer
        self.clip_model_id = clip_model_id
        self._clip_model = None
        self._clip_processor = None

    def _get_clip_score(self, image, prompt: str) -> float:
        """Compute CLIP text-image alignment score."""
        try:
            if self._clip_model is None:
                from transformers import CLIPProcessor, CLIPModel
                self._clip_model = CLIPModel.from_pretrained(self.clip_model_id)
                self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
                self._clip_model.eval()
                if torch.cuda.is_available():
                    self._clip_model = self._clip_model.cuda()

            inputs = self._clip_processor(text=[prompt], images=image, return_tensors="pt", truncation=True,
                                         max_length=self._clip_model.config.text_config.max_position_embeddings)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
            score = outputs.logits_per_image.item() / 100.0
            return min(max(score, 0.0), 1.0)
        except Exception as error:
            raise RuntimeError("CLIP alignment scoring failed") from error

    def __call__(self, image, prompt: str = "") -> Dict[str, float]:
        evasion_scores = [dr(image) for dr in self.detector_rewards]
        mean_evasion = float(np.mean(evasion_scores))

        clip_score = self._get_clip_score(image, prompt) if prompt else 0.5
        aesthetic_score = (
            float(self.aesthetic_scorer(image, prompt)) if self.aesthetic_scorer is not None else 0.0
        )

        total = (
            self.weights["evasion"] * mean_evasion
            + self.weights["clip"] * clip_score
            + self.weights["aesthetic"] * aesthetic_score
        )

        return {
            "total": total,
            "evasion": mean_evasion,
            "clip": clip_score,
            "aesthetic": aesthetic_score,
            "per_detector": {dr.name: s for dr, s in zip(self.detector_rewards, evasion_scores)},
        }
