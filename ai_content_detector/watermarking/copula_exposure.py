"""Conditional copula-race probabilities for an output-consistent text API."""

import math

import torch


@torch.no_grad()
def conditional_race_probabilities(probabilities, keyed_gaussians, rho, draws=256, generator=None):
    """Rao-Blackwell estimates, normalized after integrating each lane's own noise."""
    if probabilities.ndim != 1 or probabilities.shape != keyed_gaussians.shape or not len(probabilities):
        raise ValueError("Expected aligned nonempty candidate vectors")
    if not 0 <= rho <= 1 or not isinstance(draws, int) or isinstance(draws, bool) or draws < 1:
        raise ValueError("Invalid copula strength or integration budget")
    p, g = probabilities.double(), keyed_gaussians.double()
    if not torch.isfinite(p).all() or not torch.isfinite(g).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Nonfinite or invalid candidate distribution")
    p = p / p.sum()
    active = p > 0
    values, gaussians = p[active], g[active]
    if len(values) == 1 or rho == 0:
        return p, {"raw_sum": 1., "draws": 0, "estimator": "exact"}
    if rho == 1:
        winner = (-torch.special.log_ndtr(gaussians) / values).argmin()
        q = torch.zeros_like(values)
        q[winner] = 1.
        raw_sum = 1.
    else:
        sigma = math.sqrt(1 - rho * rho)
        noise = torch.randn((draws, len(values)), device=p.device, dtype=torch.float64, generator=generator)
        clocks = -torch.special.log_ndtr(rho * gaussians + sigma * noise) / values
        times, indices = clocks.topk(2, dim=-1, largest=False)
        other_minimum = times[:, :1].expand_as(clocks).clone()
        other_minimum.scatter_(1, indices[:, :1], times[:, 1:2])
        threshold = torch.special.ndtri(torch.exp(-values * other_minimum))
        q = torch.special.ndtr((rho * gaussians - threshold) / sigma).mean(0)
        raw_sum = float(q.sum())
        if not math.isfinite(raw_sum) or raw_sum <= 0:
            raise FloatingPointError("Conditional race integral is numerically empty")
        q = q / raw_sum
    result = torch.zeros_like(p)
    result[active] = q
    return result, {"raw_sum": raw_sum, "draws": draws if rho < 1 else 0,
                    "estimator": "leave-one-lane-out conditional expectation; normalization introduces finite-draw bias"}
