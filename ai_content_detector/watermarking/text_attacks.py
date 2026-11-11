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


def source_ngram_completions(tokens, order=4):
    if order < 2:
        raise ValueError("Source blocking requires an n-gram order of at least two")
    result = {}
    for start in range(len(tokens) - order + 1):
        context = tuple(tokens[start:start + order - 1])
        result.setdefault(context, set()).add(tokens[start + order - 1])
    return result


def constrained_api_choice(target, forbidden, reference=None, lower=None, upper=None, alpha=1., generator=None):
    """Block source completions after mixing; confidence uses the unmodified API."""
    probabilities = target.probabilities / target.probabilities.sum(-1, keepdim=True)
    ids = target.ids
    weight = None
    if reference is not None:
        if target.vocabulary_size != reference.vocabulary_size:
            raise ValueError("Constrained smoothing requires identical vocabularies")
        weight = target_weight(target, lower, upper, alpha)
        q = reference.probabilities / reference.probabilities.sum(-1, keepdim=True)
        probabilities = torch.cat((weight[:, None] * probabilities, (1-weight[:, None]) * q), -1)
        ids = torch.cat((ids, reference.ids), -1)
    blocked = torch.tensor(sorted(forbidden), device=ids.device, dtype=ids.dtype)
    allowed = ~(ids[..., None] == blocked).any(-1)
    constrained = probabilities * allowed
    fallback = constrained.sum(-1) <= 0
    probabilities = torch.where(fallback[:, None], probabilities, constrained)
    choice = ids.gather(-1, torch.multinomial(probabilities, 1, generator=generator)).squeeze(-1)
    return choice, weight, fallback


def ratio_api_choice(target, reference_probabilities, forbidden=(), exponent=2., generator=None):
    """Tilt the exposed support by a surrogate-to-API probability ratio."""
    if not exponent >= 1 or not torch.isfinite(torch.tensor(exponent)):
        raise ValueError("A finite ratio exponent of at least one is required")
    q, p = target.probabilities.double(), reference_probabilities.double()
    if q.shape != p.shape or q.shape != target.ids.shape or q.ndim != 2:
        raise ValueError("Surrogate probabilities must match the exposed token IDs")
    if not torch.isfinite(q).all() or not torch.isfinite(p).all() or (q < 0).any() or (p < 0).any():
        raise ValueError("Invalid ratio probabilities")
    supported = (q > 0) & (p > 0)
    if not supported.any(-1).all():
        raise ValueError("Ratio attack has no common positive support")
    blocked = torch.tensor(sorted(forbidden), device=target.ids.device, dtype=target.ids.dtype)
    allowed = supported & ~(target.ids[..., None] == blocked).any(-1)
    fallback = ~allowed.any(-1)
    allowed = torch.where(fallback[:, None], supported, allowed)
    logits = (1-exponent)*q.clamp_min(1.e-12).log() + exponent*p.clamp_min(1.e-12).log()
    probabilities = logits.masked_fill(~allowed, -torch.inf).softmax(-1)
    choice = target.ids.gather(-1, torch.multinomial(probabilities, 1, generator=generator)).squeeze(-1)
    return choice, fallback


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
