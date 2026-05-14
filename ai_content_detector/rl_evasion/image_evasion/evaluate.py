"""Evaluation metrics for image evasion experiments."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_image_evasion(
    images: list,
    prompts: List[str],
    reward_fn,
) -> Dict[str, float]:
    """Evaluate image evasion across multiple metrics.

    Args:
        images: PIL Images produced by the evasion model.
        prompts: Prompts used to generate them.
        reward_fn: CompositeImageReward instance.

    Returns:
        Dict with mean_total, mean_evasion, mean_clip, attack_success_rate,
        per_detector breakdown.
    """
    all_rewards = {"total": [], "evasion": [], "clip": [], "aesthetic": []}
    per_detector: Dict[str, List[float]] = {}

    for image, prompt in zip(images, prompts):
        result = reward_fn(image, prompt)
        for k in all_rewards:
            all_rewards[k].append(result.get(k, 0.5))
        for det_name, score in result.get("per_detector", {}).items():
            per_detector.setdefault(det_name, []).append(score)

    metrics = {f"mean_{k}": float(np.mean(v)) for k, v in all_rewards.items() if v}

    # Attack success rate: evasion > 0.5
    if all_rewards["evasion"]:
        metrics["attack_success_rate"] = float(np.mean([e > 0.5 for e in all_rewards["evasion"]]))

    # Per-detector
    for det_name, scores in per_detector.items():
        metrics[f"evasion_{det_name}"] = float(np.mean(scores))

    metrics["num_samples"] = len(images)
    return metrics


def print_image_evaluation_report(metrics: Dict[str, float], title: str = "Image Evasion Report"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Samples evaluated:    {metrics.get('num_samples', 'N/A')}")
    print(f"  Mean total reward:    {metrics.get('mean_total', 0):.4f}")
    print(f"  Mean evasion:         {metrics.get('mean_evasion', 0):.4f}")
    print(f"  Mean CLIP score:      {metrics.get('mean_clip', 0):.4f}")
    print(f"  Attack success rate:  {metrics.get('attack_success_rate', 0):.1%}")

    det_keys = [k for k in metrics if k.startswith("evasion_")]
    if det_keys:
        print(f"\n  Per-detector evasion:")
        for k in det_keys:
            det_name = k.replace("evasion_", "")
            print(f"    {det_name:30s}: {metrics[k]:.4f}")

    print(f"{'=' * 60}\n")
