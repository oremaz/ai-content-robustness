import numpy as np
import pytest

from ai_content_detector.watermarking.survival import survival_gains


def test_survival_filter_matches_diagonal_closed_form_and_preserves_delta_energy():
    curvature = {"slow": .1, "fast": 2.}
    energy = {"slow": 4., "fast": 1.}
    gain, receipt = survival_gains(curvature, energy, horizon=2.)
    h = np.array([.1, 2.]) / ((4 * .1 + 2) / 5)
    expected = np.exp(-2 * h) / (h + 1.)
    expected *= np.sqrt(5 / np.dot([4, 1], expected ** 2))
    np.testing.assert_allclose([gain["slow"], gain["fast"]], expected, rtol=1e-12)
    assert sum(energy[k] * gain[k] ** 2 for k in energy) == pytest.approx(5.)
    assert gain["slow"] > gain["fast"]
    assert receipt["delta_energy"] == 5.


def test_filter_is_invariant_to_fisher_units_and_handles_zero_delta_blocks():
    h, e = {"a": .1, "b": 3., "unused": 0.}, {"a": 2., "b": 4., "unused": 0.}
    first, _ = survival_gains(h, e, 1.)
    second, _ = survival_gains({k: v * 1e-10 for k, v in h.items()}, e, 1.)
    np.testing.assert_allclose(list(first.values()), list(second.values()))
    assert survival_gains({"a": 0.}, {"a": 1.}, 3.)[0] == {"a": 1.}
    with pytest.raises(ValueError):
        survival_gains({"a": -1.}, {"a": 1.}, 1.)
    with pytest.raises(ValueError):
        survival_gains({"a": 1.}, {"a": 0.}, 1.)
