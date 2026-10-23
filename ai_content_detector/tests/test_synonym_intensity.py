import csv
import json
from types import SimpleNamespace

import pytest

from scripts import synonym_intensity as intensity
from scripts import synonym_robustness as synonyms
from ai_content_detector.rl_evasion.benchmarking.synonym_robustness import apply_proposal
from ai_content_detector.rl_evasion.defender_comparison.protocol import digest


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(intensity, "ROOT", tmp_path)
    monkeypatch.setattr(intensity, "OUTPUT", tmp_path / "results")
    monkeypatch.setattr(intensity, "inputs", lambda: ({"fixed": True}, {}))
    intensity.write_record(intensity.OUTPUT / "architecture_races.json", {"jobs": {"roberta": {"job_id": "10"}}})
    calls = []
    monkeypatch.setattr(intensity.subprocess, "check_output", lambda command, **kwargs: calls.append(command) or str(100+len(calls)))
    return intensity.submit(), calls


def test_preparation_and_scoring_are_separate_held_development_jobs(tmp_path, monkeypatch):
    path, calls = fixture(tmp_path, monkeypatch)
    manifest = json.loads(path.read_text())
    assert manifest["sample_indices"] == list(range(100))
    assert manifest["conditions"] == {"p10": 0.1, "p25": 0.25}
    assert manifest["detector_names"] == ["roberta", "decoder_classifier"]
    assert manifest["score_held"] and manifest["review_required"]
    assert manifest["capacity_blocker"] == "102"
    assert all("--hold" in call and "--qos=qos_gpu_h100-dev" in call and "--time=02:00:00" in call for call in calls)
    assert "--dependency=afterok:101" in calls[1]


def test_capacity_is_reserved_until_scoring_finishes(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(synonyms.subprocess, "check_output", lambda *args, **kwargs: "PENDING\n")
    calls = []
    monkeypatch.setattr(synonyms.subprocess, "run", lambda command, **kwargs: calls.append(command))
    intensity.release(path)
    assert ["scontrol", "update", "JobId=10", "Dependency=afterany:102"] in calls
    assert ["scontrol", "release", "101"] in calls
    assert ["scontrol", "release", "102"] not in calls


def review_fixture(directory):
    review = {"reviewer": "fixture", "review_before_new_scoring": True, "conditions": {}, "source_variants_sha256": {}}
    for name, rate in intensity.CONDITIONS.items():
        specification = {"budget": int(rate*20), "eligible_content_words": 20, "candidates": [
            {"index": 0, "start": 0, "end": 3, "original": "big", "choices": ["large"]},
            {"index": 1, "start": 4, "end": 9, "original": "house", "choices": ["home"]}]}
        proposed = json.dumps({"edits": [{"index": 0, "replacement": "large"}, {"index": 1, "replacement": "home"}]})
        raw = [{"index": i, "original": "big house", "sample_id": str(i), **specification,
                **apply_proposal("big house", specification, proposed)} for i in range(100)]
        intensity.write_record(directory/name/"variants.json", raw)
        review["conditions"][name] = [{"index": i, "accepted": [0],
            "rejected": [{"index": 1, "reason": "context mismatch"}], "coherent_after_review": True} for i in range(100)]
        review["source_variants_sha256"][name] = digest(raw)
    intensity.write_record(directory/"review.json", review)
    return review


def test_coherence_review_removes_bad_edits_before_freezing(tmp_path):
    review_fixture(tmp_path)
    result = intensity.reviewed_variants({}, tmp_path)
    assert all(len(rows) == 100 for rows in result.values())
    assert all(row["text"] == "large house" and len(row["edits"]) == 1 for rows in result.values() for row in rows)
    assert all(len(row["proposed_edits"]) == 2 for rows in result.values() for row in rows)


def test_review_can_correct_only_previously_rejected_proposal_positions(tmp_path):
    review = review_fixture(tmp_path)
    review["conditions"]["p25"][0]["corrections"] = [{"index": 1, "replacement": "home", "reason": "suitable contextual alternative"}]
    intensity.write_record(tmp_path/"review.json", review)
    assert intensity.reviewed_variants({}, tmp_path)["p25"][0]["text"] == "large home"
    review["conditions"]["p25"][0]["corrections"][0]["index"] = 9
    intensity.write_record(tmp_path/"review.json", review)
    with pytest.raises(ValueError):
        intensity.reviewed_variants({}, tmp_path)


@pytest.mark.parametrize("error", ["missing_row", "missing_edit", "wrong_source", "not_coherent"])
def test_incomplete_or_changed_reviews_cannot_pass(tmp_path, error):
    review = review_fixture(tmp_path)
    if error == "missing_row":
        review["conditions"]["p25"].pop()
    elif error == "missing_edit":
        review["conditions"]["p25"][0]["rejected"] = []
    elif error == "wrong_source":
        review["source_variants_sha256"]["p25"] = "changed"
    else:
        review["conditions"]["p25"][0]["coherent_after_review"] = False
    intensity.write_record(tmp_path/"review.json", review)
    with pytest.raises(ValueError):
        intensity.reviewed_variants({}, tmp_path)


def test_only_reviewed_score_job_is_released(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path, monkeypatch)
    review_fixture(path.parent)
    intensity.write_record(path.parent/"prepare_receipt.json", {"status": "completed"})
    calls = []
    monkeypatch.setattr(intensity.subprocess, "run", lambda command, **kwargs: calls.append(command))
    intensity.release_score(path)
    assert calls == [["scontrol", "release", "102"]]
    assert not json.loads(path.read_text())["score_held"]
    assert (path.parent/"review_receipt.json").exists()
    review = json.loads((path.parent/"review.json").read_text())
    review["conditions"]["p25"][0]["accepted"] = []
    review["conditions"]["p25"][0]["rejected"].append({"index": 0, "reason": "later change"})
    intensity.write_record(path.parent/"review.json", review)
    with pytest.raises(ValueError, match="already frozen"):
        intensity.freeze_review(path)


def test_selected_100_rows_use_correct_human_baseline_and_only_two_detectors(tmp_path, monkeypatch):
    from ai_content_detector.rl_evasion.defender_comparison import workflow, defenders, protocol
    from ai_content_detector.rl_evasion.text_evasion import rewards

    names = intensity.DETECTORS
    pairs = [{"sample_id": str(i)} for i in range(100)]
    baseline = {"eligible_ids": [str(i) for i in range(500)],
                "scores": {name: [0.99]*500 + [0.1]*500 for name in names}}
    records = [{"sample_id": str(i), "original": "big house", "text": "large house" if i == 0 else "big house",
                "edits": [{}] if i == 0 else [], "eligible_content_words": 20, "rejections": []} for i in range(100)]
    monkeypatch.setattr(rewards, "SemanticSimilarityReward", lambda **kwargs: lambda a, b: 0.99)
    monkeypatch.setattr(workflow, "warm_checkpoints", lambda: dict.fromkeys(names, "fixture"))

    def pool(specs, *args, **kwargs):
        assert list(specs) == names and set(kwargs["checkpoints"]) == set(names)
        return SimpleNamespace(clear=lambda: None)

    monkeypatch.setattr(defenders, "DefenderPool", pool)
    monkeypatch.setattr(protocol, "score_checkpointed", lambda pool, texts, directory: {name: [0.01]+[0.99]*99 for name in names})
    manifest = {"sample_indices": list(range(100)), "detector_names": names, "include_ensemble": False}
    result = synonyms.measure_variants(manifest, tmp_path, pairs, baseline,
                                     {"specifications": dict.fromkeys(names, {})}, records, SimpleNamespace(requested=False))
    assert len(result["records"]) == 2
    assert all(row["human_false_positive_rate"] == 0 and row["ai_detection_after"] == 0.99 for row in result["records"])
    assert len(list(csv.DictReader((tmp_path/"paired_scores.csv").open()))) == 200
