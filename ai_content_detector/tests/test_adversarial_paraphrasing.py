from __future__ import annotations

import torch

from ai_content_detector.rl_evasion.benchmarking.baselines import AdversarialParaphrasingBaseline


def test_candidate_filter_applies_paper_top_p_and_top_k():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.1, 0.07, 0.03]))
    candidates = AdversarialParaphrasingBaseline._candidate_token_ids(
        logits, top_p=0.79, top_k=2,
    )
    assert candidates.tolist() == [0, 1]


def test_paper_defaults():
    baseline = AdversarialParaphrasingBaseline(lambda text: 0.5)
    assert baseline._top_p == 0.99
    assert baseline._top_k == 50
