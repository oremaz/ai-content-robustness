"""Check the support optimizer against exhaustive finite selection problems."""

from itertools import combinations

import pytest
import torch

from ai_content_detector.watermarking.exposure_support import maximin_support, evidence_lower_bound


def test_support_matches_exhaustive_maximin_optimization():
    generator = torch.Generator().manual_seed(2026110713)
    for _ in range(20):
        p = torch.rand(7, generator=generator, dtype=torch.float64)
        p /= p.sum()
        scores = torch.rand(7, generator=generator, dtype=torch.float64)*4
        target = .85*p.topk(3).values.sum()
        feasible = [choice for size in range(1, 4) for choice in combinations(range(7), size)
                    if p[list(choice)].sum() >= target-1.e-12]
        optimum = max(float(scores[list(choice)].min()) for choice in feasible)
        original_p, original_scores = p.clone(), scores.clone()
        ids, receipt = maximin_support(p, scores, k=3, coverage_loss=.15)
        assert len(ids) <= 3 and p[ids].sum() >= target-1.e-12
        assert receipt["score_floor"] == pytest.approx(optimum)
        assert torch.equal(p, original_p) and torch.equal(scores, original_scores)


def test_support_handles_ties_zero_mass_and_zero_coverage_loss():
    p = torch.tensor([.6, .3, .1, 0.], dtype=torch.float64)
    scores = torch.tensor([1., 3., 3., 100.], dtype=torch.float64)
    ids, receipt = maximin_support(p, scores, k=2, coverage_loss=0.)
    assert set(ids.tolist()) == {0, 1}
    assert receipt["retained_mass"] == pytest.approx(.9)
    ids, receipt = maximin_support(p, scores, k=2, coverage_loss=.6)
    assert set(ids.tolist()) == {1, 2}
    assert receipt["score_floor"] == 3.


def test_singleton_is_not_presented_as_hardened_evidence():
    ids, receipt = maximin_support(torch.tensor([1., 0.]), torch.tensor([.2, 10.]), k=10)
    assert ids.tolist() == [0] and receipt["score_floor"] == pytest.approx(.2)
    assert receipt["relative_mass_loss"] == 0.


def test_pruned_topk_fast_path_has_exact_tv_and_optimal_floor():
    generator = torch.Generator().manual_seed(2026110801)
    for _ in range(20):
        p = torch.rand(7, generator=generator, dtype=torch.float64)
        p = p / p.sum() * .7
        scores = torch.randint(0, 4, (7,), generator=generator).double()
        ids, receipt = maximin_support(p, scores, k=10, coverage_loss=.2)
        feasible = [choice for size in range(1, 8) for choice in combinations(range(7), size)
                    if p[list(choice)].sum() >= .8*p.sum()-1.e-12]
        assert receipt["score_floor"] == max(float(scores[list(choice)].min()) for choice in feasible)
        restricted = torch.zeros_like(p)
        restricted[ids] = p[ids]/p[ids].sum()
        tv = .5*(restricted-p/p.sum()).abs().sum()
        assert float(tv) == pytest.approx(receipt["relative_mass_loss"], abs=1.e-12)
        assert float(tv) <= .2+1.e-12


@pytest.mark.parametrize("p,s,k,loss", [([0.,0.],[1.,2.],1,.1), ([1.],[float('nan')],1,.1),
                                      ([-1.,2.],[1.,2.],1,.1), ([1.],[1.],0,.1), ([1.],[1.],1,1.)])
def test_invalid_support_constraints_are_rejected(p,s,k,loss):
    with pytest.raises(ValueError):
        maximin_support(torch.tensor(p), torch.tensor(s), k, loss)


def test_evidence_bound_excludes_prompt_windows_repeats_and_external_choices():
    tokens=[1,2,3,4,1,2,3,4]
    result=evidence_lower_bound(tokens,[100.,100.,100.,2.,None,3.,4.,100.],
        [True,True,True,True,False,True,True,True],tolerance=0.)
    assert result["scored_pairs"]==4
    assert result["guarded_scored_pairs"]==3
    assert result["score_lower_bound"]==9.
    with pytest.raises(ValueError,match="requires an evidence floor"):
        evidence_lower_bound([1,2,3,4],[None]*4,[False,False,False,True])
