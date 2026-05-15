"""Evaluation metrics for text evasion experiments."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_text_evasion(
    generated_texts: List[str],
    source_texts: List[str],
    reward_fn,
) -> Dict[str, float]:
    """Evaluate text evasion across multiple metrics.

    Args:
        generated_texts: Texts produced by the evasion model.
        source_texts: Original source texts (prompts).
        reward_fn: CompositeReward instance for scoring.

    Returns:
        Dict with metrics: mean_reward, mean_evasion, mean_semantic,
        attack_success_rate, per_detector scores.
    """
    all_rewards = []
    all_evasion = []
    all_semantic = []
    per_detector_scores: Dict[str, List[float]] = {}

    for source, generated in zip(source_texts, generated_texts):
        result = reward_fn(source, generated)
        all_rewards.append(result["total"])
        all_evasion.append(result["evasion"])
        all_semantic.append(result["semantic"])

        for det_name, score in result.get("per_detector", {}).items():
            if det_name not in per_detector_scores:
                per_detector_scores[det_name] = []
            per_detector_scores[det_name].append(score)

    # Attack success rate: fraction of samples where evasion > 0.5
    # (i.e., detector score < 0.5, meaning the detector would label as human)
    attack_success = np.mean([e > 0.5 for e in all_evasion])

    metrics = {
        "mean_reward": float(np.mean(all_rewards)),
        "mean_evasion": float(np.mean(all_evasion)),
        "mean_semantic_similarity": float(np.mean(all_semantic)),
        "attack_success_rate": float(attack_success),
        "num_samples": len(generated_texts),
    }

    # Per-detector metrics
    for det_name, scores in per_detector_scores.items():
        metrics[f"evasion_{det_name}"] = float(np.mean(scores))

    # Mean detector AI-score (lower = better evasion)
    detector_scores = [1.0 - e for e in all_evasion]  # convert back to detector score
    if detector_scores:
        metrics["mean_detector_ai_score"] = float(np.mean(detector_scores))

    return metrics


def compute_tpr_at_fpr(
    detector_scores: List[float],
    labels: List[int],
    fpr_target: float,
) -> float:
    """Compute TPR at a specific FPR operating point.

    Args:
        detector_scores: AI probability scores from detector.
        labels: Ground truth (1 = AI, 0 = human).
        fpr_target: Target false positive rate.

    Returns:
        TPR at the given FPR operating point.
    """
    from sklearn.metrics import roc_curve

    scores = np.array(detector_scores)
    labels_arr = np.array(labels)

    fpr, tpr, _ = roc_curve(labels_arr, scores)

    # Linear interpolation of TPR at the target FPR (the raw ROC grid rarely
    # lands exactly on e.g. 0.01, so nearest-neighbor lookup gives biased values
    # at the low-FPR operating points papers actually care about).
    idx = int(np.searchsorted(fpr, fpr_target))
    if idx == 0:
        return float(tpr[0])
    if idx >= len(fpr):
        return float(tpr[-1])
    lo, hi = float(fpr[idx - 1]), float(fpr[idx])
    if hi == lo:
        return float(tpr[idx])
    w = (fpr_target - lo) / (hi - lo)
    return float((1.0 - w) * tpr[idx - 1] + w * tpr[idx])


def compute_auroc(
    detector_scores: List[float],
    labels: List[int],
) -> float:
    """Compute AUROC for a detector."""
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(labels, detector_scores))
    except ValueError:
        return 0.5


def print_evaluation_report(metrics: Dict[str, float], title: str = "Evaluation Report"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Samples evaluated:        {metrics.get('num_samples', 'N/A')}")
    print(f"  Mean reward:              {metrics.get('mean_reward', 0):.4f}")
    print(f"  Mean evasion score:       {metrics.get('mean_evasion', 0):.4f}")
    print(f"  Mean semantic similarity: {metrics.get('mean_semantic_similarity', 0):.4f}")
    print(f"  Attack success rate:      {metrics.get('attack_success_rate', 0):.1%}")
    print(f"  Mean detector AI score:   {metrics.get('mean_detector_ai_score', 0):.4f}")

    # Per-detector
    det_keys = [k for k in metrics if k.startswith("evasion_")]
    if det_keys:
        print(f"\n  Per-detector evasion scores:")
        for k in det_keys:
            det_name = k.replace("evasion_", "")
            print(f"    {det_name:30s}: {metrics[k]:.4f}")

    print(f"{'=' * 60}\n")
