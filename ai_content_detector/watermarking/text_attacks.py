"""Watermark-agnostic BIRA and probability smoothing attack primitives."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TopKResponse:
    ids: torch.Tensor
    probabilities: torch.Tensor
    vocabulary_size: int


def confidence(response):
    p = response.probabilities.float()
    tail_size = response.vocabulary_size - p.shape[-1]
    if tail_size < 0:
        raise ValueError("Top-k exceeds vocabulary")
    tail = (1 - p.sum(-1)).clamp_min(0).square() / tail_size if tail_size else 0
    return p.square().sum(-1) + tail


def target_weight(response, lower, upper, alpha=1.):
    if not 0 <= lower < upper <= 1 or alpha <= 0:
        raise ValueError("Invalid confidence bounds or smoothing exponent")
    return ((confidence(response) - lower) / (upper - lower)).clamp(0, 1).pow(alpha)


def resample(response, generator=None):
    return response.ids.gather(-1, torch.multinomial(response.probabilities, 1, generator=generator)).squeeze(-1)


def smooth(target, reference, lower, upper, alpha=1., generator=None):
    if target.vocabulary_size != reference.vocabulary_size:
        raise ValueError("Smoothing requires identical token vocabularies")
    weight = target_weight(target, lower, upper, alpha)
    p = target.probabilities / target.probabilities.sum(-1, keepdim=True)
    q = reference.probabilities / reference.probabilities.sum(-1, keepdim=True)
    # Repeated IDs in the two supports contribute additive probability mass.
    probabilities = torch.cat((weight[:, None] * p, (1-weight[:, None]) * q), -1)
    ids = torch.cat((target.ids, reference.ids), -1)
    return ids.gather(-1, torch.multinomial(probabilities, 1, generator=generator)).squeeze(-1), weight


def suppression_from_logits(logits, ids, percentile=.5):
    if not 0 <= percentile < 1 or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Expected one document and a percentile in [0, 1)")
    if ids.shape[1] < 2:
        return ids.new_empty(0)
    targets = ids[0, 1:]
    losses = torch.cat([F.cross_entropy(logits[0, start:min(start+64, len(targets))].float(), targets[start:start+64],
                                       reduction="none") for start in range(0, len(targets), 64)])
    return targets[losses >= torch.quantile(losses, percentile)].unique()


def degenerated(ids, window=450, threshold=.25):
    recent = ids[-window:]
    return bool(recent) and len(set(recent)) / len(recent) < threshold


class SuppressTokens:
    def __init__(self, sets, betas):
        self.sets, self.betas = sets, betas

    def __call__(self, input_ids, scores):
        for i, (tokens, beta) in enumerate(zip(self.sets, self.betas)):
            scores[i, tokens] += beta
        return scores
