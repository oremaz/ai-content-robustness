"""Candidate post-distillation filtering of an existing watermark task vector."""

from __future__ import annotations

import math

import numpy as np


def survival_gains(curvature, energy, horizon, ridge=1.):
    """Block-diagonal Fisher surrogate, normalized to the original delta norm."""
    if set(curvature) != set(energy) or not energy:
        raise ValueError("Curvature and watermark energy blocks must match")
    if not math.isfinite(horizon) or horizon < 0 or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Horizon must be nonnegative and ridge strictly positive")
    names = sorted(energy)
    h = np.array([curvature[k] for k in names], dtype=np.float64)
    e = np.array([energy[k] for k in names], dtype=np.float64)
    if not np.isfinite(h).all() or not np.isfinite(e).all() or (h < 0).any() or (e < 0).any():
        raise ValueError("Curvature and energy must be finite and nonnegative")
    if e.sum() <= 0:
        raise ValueError("The supplied model has no watermark task vector")
    scale = float(np.dot(h, e) / e.sum())
    h = h / scale if scale > 0 else h
    log_gain = -horizon * h - np.log(h + ridge)
    active = e > 0
    log_gain -= np.max(log_gain[active])
    gain = np.zeros_like(h)
    gain[active] = np.exp(log_gain[active])
    norm = float(np.sqrt(np.dot(e, gain * gain) / e.sum()))
    if not math.isfinite(norm) or norm <= 0:
        raise FloatingPointError("Watermark survival filter has degenerate energy")
    gain /= norm
    gain[~active] = 1.
    return dict(zip(names, gain.tolist())), {"curvature_scale": scale, "delta_energy": float(e.sum()),
                                            "gain_min": float(gain.min()), "gain_max": float(gain.max())}
