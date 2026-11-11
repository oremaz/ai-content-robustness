import math

import pytest
import torch

from ai_content_detector.watermarking.copula_exposure import conditional_race_probabilities


def test_two_equal_lanes_have_analytic_gaussian_comparison_probability():
    p = torch.tensor([.5, .5, 0.], dtype=torch.float64)
    g = torch.tensor([.6, -.3, 20.], dtype=torch.float64)
    rho = .7
    q, receipt = conditional_race_probabilities(p, g, rho, draws=32768,
                                               generator=torch.Generator().manual_seed(1909))
    expected = torch.special.ndtr(rho * (g[0] - g[1]) / math.sqrt(2 * (1 - rho**2)))
    assert abs(float(q[0] - expected)) < .006
    assert q[2] == 0 and abs(float(q.sum()) - 1) < 1.e-12
    assert abs(receipt['raw_sum'] - 1) < .01


def test_independent_and_deterministic_limits_and_invalid_inputs():
    p = torch.tensor([.2, .8]);g = torch.tensor([2., -2.])
    q, _ = conditional_race_probabilities(p, g, 0.)
    assert torch.allclose(q, p.double())
    q, _ = conditional_race_probabilities(p, g, 1.)
    assert torch.equal(q, torch.tensor([1., 0.], dtype=torch.float64))
    for invalid in [torch.tensor([0., 0.]), torch.tensor([float('nan'), 1.])]:
        with pytest.raises(ValueError):conditional_race_probabilities(invalid, g, .9)
    with pytest.raises(ValueError):conditional_race_probabilities(p, g, .9, draws=0)
