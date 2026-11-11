import math

import pytest

from ai_content_detector.watermarking.preservation import joint_outcomes, numeric_gate


def test_joint_denominators_exclude_undetected_originals_and_retain_failed_attacks():
    original = {"a": True, "b": True, "c": True, "d": False}
    attacked = {"a": False, "b": False, "c": True, "d": False}
    quality = {key: {"passed": key != "b", "failures": ["changed_fact"] if key == "b" else []}
               for key in original}
    result = joint_outcomes(original, attacked, quality)
    assert result["raw_removal"] == {"count": 2, "n": 3, "rate": 2 / 3}
    assert result["joint_removal"] == {"count": 1, "n": 3, "rate": 1 / 3}
    assert result["removal_given_preservation"] == {"count": 1, "n": 2, "rate": .5}
    assert result["preservation"] == {"count": 3, "n": 4, "rate": .75}
    assert result["joint_success_ids"] == ["a"]
    assert result["preservation_failure_reasons"] == {"changed_fact": 1}


def test_quality_constraints_reject_missing_and_nonfinite_values_without_deleting_rows():
    bounds = {"lpips": [0, .2], "clip_image_cosine": [.9, 1]}
    assert numeric_gate({"lpips": .2, "clip_image_cosine": .9}, bounds)["passed"]
    for bad in [None, math.nan, math.inf, True]:
        assert not numeric_gate({"lpips": bad, "clip_image_cosine": 1.}, bounds)["passed"]
    assert not numeric_gate({"lpips": .01}, bounds)["passed"]
    with pytest.raises(ValueError):
        numeric_gate({}, {})


def test_alignment_and_undefined_conditional_rates_cannot_silently_change_denominators():
    quality = {"a": {"passed": False, "failures": ["not_preserved"]}}
    result = joint_outcomes({"a": False}, {"a": False}, quality)
    assert result["joint_removal"]["rate"] is None
    assert result["removal_given_preservation"]["rate"] is None
    with pytest.raises(ValueError, match="match exactly"):
        joint_outcomes({"a": True}, {"b": False}, quality)
    with pytest.raises(ValueError, match="disagree"):
        joint_outcomes({"a": True}, {"a": False}, {"a": {"passed": True, "failures": ["bad"]}})
