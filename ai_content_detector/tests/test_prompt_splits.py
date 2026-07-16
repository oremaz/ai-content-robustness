from __future__ import annotations

import numpy as np


def test_seeded_split_is_disjoint_and_reproducible():
    prompts = [f"prompt-{index}" for index in range(20)]

    def split(seed):
        permutation = np.random.default_rng(seed).permutation(len(prompts))
        evaluation = {prompts[index] for index in permutation[:4]}
        training = {prompts[index] for index in permutation[4:]}
        return training, evaluation

    first = split(42)
    second = split(42)
    assert first == second
    assert first[0].isdisjoint(first[1])
