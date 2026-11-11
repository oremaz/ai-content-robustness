"""A blind quality gate may select source outputs, never detector outcomes."""

import pytest

from ai_content_detector.watermarking.preservation import preserving_fallback


def test_fallback_preserves_identity_and_keeps_only_quality_passing_changes():
    originals, candidates = {"a": "source a", "b": "source b"}, {"a": "new a", "b": "new b"}
    gates = {"a": {"passed": True, "failures": []}, "b": {"passed": False, "failures": ["similarity"]}}
    selected, ids = preserving_fallback(originals, candidates, gates)
    assert selected == {"a": "new a", "b": "source b"}
    assert ids == ["b"]
    assert candidates["b"] == "new b"


def test_fallback_rejects_missing_ids_and_inconsistent_quality_records():
    with pytest.raises(ValueError, match="IDs must match"):
        preserving_fallback({"a": 1}, {"b": 2}, {"a": {"passed": True, "failures": []}})
    with pytest.raises(ValueError, match="disagree"):
        preserving_fallback({"a": 1}, {"a": 2}, {"a": {"passed": True, "failures": ["similarity"]}})
