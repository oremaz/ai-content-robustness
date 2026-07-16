from __future__ import annotations

import random

import numpy as np
import torch

from ai_content_detector.rl_evasion.config import seed_everything


def test_seed_everything_replays_python_numpy_and_torch():
    seed_everything(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second
