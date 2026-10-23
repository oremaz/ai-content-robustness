import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_content_detector.rl_evasion.benchmarking.synonym_robustness import (
    apply_proposal, candidates, paired_metrics, request_text,
)
from scripts import synonym_robustness as job


def test_candidates_limit_edits_and_protect_names_numbers_and_inflection():
    words = [SimpleNamespace(i=i, idx=4*i, text="big", lemma_="big", pos_="ADJ",
                             is_alpha=True, is_stop=False, ent_type_="") for i in range(40)]
    words += [SimpleNamespace(i=40, idx=160, text="London", lemma_="London", pos_="NOUN",
                              is_alpha=True, is_stop=False, ent_type_="GPE"),
              SimpleNamespace(i=41, idx=167, text="42", lemma_="42", pos_="NUM",
                              is_alpha=False, is_stop=False, ent_type_=""),
              SimpleNamespace(i=42, idx=170, text="houses", lemma_="house", pos_="NOUN",
                              is_alpha=True, is_stop=False, ent_type_="")]
    wordnet = SimpleNamespace(synsets=lambda word, pos: [SimpleNamespace(lemmas=lambda: [
        SimpleNamespace(name=lambda: "large"), SimpleNamespace(name=lambda: "very_big")])])
    first = candidates(words, wordnet, 0)
    assert first == candidates(words, wordnet, 0)
    assert first["eligible_content_words"] == 41 and first["budget"] == 2
    assert len(first["candidates"]) == 6
    assert all(row["index"] < 40 and row["choices"] == ["large"] for row in first["candidates"])


def specification():
    return {"budget": 1, "candidates": [
        {"index": 0, "start": 0, "end": 3, "original": "Big", "choices": ["large"]},
        {"index": 1, "start": 5, "end": 10, "original": "house", "choices": ["home"]}]}


def test_proposals_preserve_formatting_and_enforce_budget_without_feedback():
    text = "Big  house,\nLondon 42."
    response = json.dumps({"edits": [{"index": 0, "replacement": "large"},
                                      {"index": 0, "replacement": "large"},
                                      {"index": 1, "replacement": "home"},
                                      {"index": 99, "replacement": "Paris"}]})
    result = apply_proposal(text, specification(), response)
    assert result["text"] == "Large  house,\nLondon 42."
    assert len(result["edits"]) == 1 and len(result["rejections"]) == 3
    assert "detector" not in request_text(text, specification()).lower()


@pytest.mark.parametrize("response", ["no JSON", '{"edits":null}', '{"edits":[null, {"index":true,"replacement":"large"}]}'])
def test_invalid_proposals_remain_visible_without_changes(response):
    result = apply_proposal("Big  house,", specification(), response)
    assert result["text"] == "Big  house," and not result["edits"] and result["rejections"]


def test_truncated_json_recovery_uses_only_complete_entries_and_keeps_budget():
    response = '{"edits":[{"index":0,"replacement":"large"},{"index":1,"replacement":"home"},{"index":'
    strict = apply_proposal("Big  house,", specification(), response)
    assert not strict["edits"]
    recovered = apply_proposal("Big  house,", specification(), response, recover_complete=True)
    assert recovered["text"] == "Large  house," and len(recovered["edits"]) == 1
    assert recovered["rejections"][0]["reason"] == "duplicate_or_budget"
    assert recovered["json_recovery"] == "only_complete_entries_from_truncated_edits_array"


def test_metrics_track_paired_flips_both_directions_and_unchanged_rows():
    value = paired_metrics([0.9, 0.2, 0.8], [0.1, 0.7, 0.8], [0.1, 0.1, 0.1], [True, True, False])
    assert value["num_changed"] == 2
    assert value["detected_to_undetected"] == value["undetected_to_detected"] == 1
    assert value["mean_score_delta"] == pytest.approx(-0.1)
    assert value["changed_only_mean_score_delta"] == pytest.approx(-0.15)
    assert value["flip_fraction_of_initially_detected"] == 0.5
    assert value["human_false_positive_rate"] == 0
    with pytest.raises(ValueError, match="identical"):
        paired_metrics([0.9], [], [0.1], [True])


def submission(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "ROOT", tmp_path)
    monkeypatch.setattr(job, "OUTPUT", tmp_path / "results/research_workflow")
    monkeypatch.setattr(job, "inputs", lambda: ({"fixed": True}, {"seed": 42}))
    from scripts import short_probes
    monkeypatch.setattr(short_probes, "split_audit", lambda: {"scored_prompts": 500})
    job.write_record(job.OUTPUT / "architecture_races.json", {"jobs": {"roberta": {"job_id": "10"}}})
    calls = []
    monkeypatch.setattr(job.subprocess, "check_output", lambda command, **kwargs: calls.append(command) or "101")
    return job.submit(), calls


def test_submission_uses_checkpointed_development_allocation(tmp_path, monkeypatch):
    path, calls = submission(tmp_path, monkeypatch)
    manifest = json.loads(path.read_text())
    assert manifest["held"] and manifest["rate"] == 0.05
    assert manifest["training_updates"] == 0 and not manifest["detector_feedback"]
    assert "--qos=qos_gpu_h100-dev" in calls[0] and "--time=02:00:00" in calls[0]
    assert "--requeue" in calls[0] and "--signal=USR1@600" in calls[0]


def test_capacity_handover_preserves_three_gpu_limit_and_race_settings(tmp_path, monkeypatch):
    path, _ = submission(tmp_path, monkeypatch)
    monkeypatch.setattr(job.subprocess, "check_output", lambda *args, **kwargs: "PENDING\n")
    calls = []
    monkeypatch.setattr(job.subprocess, "run", lambda command, **kwargs: calls.append(command))
    job.release(path)
    assert calls == [["scontrol", "hold", "10"],
                     ["scontrol", "update", "JobId=10", "Dependency=afterany:101"],
                     ["scontrol", "release", "101"], ["scontrol", "release", "10"]]
    assert not json.loads(path.read_text())["held"]


def test_paired_experiment_freezes_variants_reuses_human_scores_and_caches_results(tmp_path, monkeypatch):
    from ai_content_detector.rl_evasion.defender_comparison import workflow, defenders, protocol
    from ai_content_detector.rl_evasion.text_evasion import rewards

    names = ["binoculars", "fast_detect_gpt", "diveye", "roberta", "tfidf", "modernbert_large", "decoder_classifier"]
    plan = {"specifications": dict.fromkeys(names, {})}
    monkeypatch.setattr(job, "inputs", lambda: ({"fixed": True}, plan))
    monkeypatch.setattr(workflow, "OUTPUT", tmp_path / "original")
    monkeypatch.setattr(workflow, "warm_checkpoints", lambda: {})
    pairs = [{"sample_id": str(i), "ai": "big house", "eligible": True} for i in range(500)]
    job.write_record(workflow.OUTPUT / "base_pairs.json", {"evaluation": pairs})
    job.write_record(workflow.OUTPUT / "warmup/measurements.json", {
        "eligible_ids": [str(i) for i in range(500)], "scores": {name: [0.9]*500 + [0.1]*500 for name in names}})
    records = [{"sample_id": str(i), "original": "big house", "text": "large house" if i == 0 else "big house",
                "eligible_content_words": 20, "edits": [{"original": "big", "replacement": "large"}] if i == 0 else [],
                "rejections": []} for i in range(500)]
    monkeypatch.setattr(job, "variants", lambda *args: records)
    monkeypatch.setattr(rewards, "SemanticSimilarityReward", lambda **kwargs: lambda a, b: 0.99)
    monkeypatch.setattr(defenders, "DefenderPool", lambda *args, **kwargs: SimpleNamespace(clear=lambda: None))
    directory = tmp_path / "run"
    directory.mkdir()

    def score(pool, texts, output):
        assert len(texts) == 500 and texts[0] == "large house"
        assert json.loads((directory / "frozen_variants.json").read_text())["detector_feedback"] is False
        return {name: [0.1] + [0.9]*499 for name in names}

    monkeypatch.setattr(protocol, "score_checkpointed", score)
    manifest = {"inputs": {"fixed": True}, "baseline_plan": plan, "rate": 0.05,
                "training_updates": 0, "detector_feedback": False}
    result = job.experiment(manifest, directory, SimpleNamespace(requested=False))
    assert len(result["records"]) == 8 and result["num_changed"] == 1
    assert all(row["detected_to_undetected"] == 1 for row in result["records"])
    assert len(list(csv.DictReader((directory / "paired_scores.csv").open()))) == 4000
    monkeypatch.setattr(job, "variants", lambda *args: pytest.fail("Repeated completed edits"))
    assert job.experiment(manifest, directory, SimpleNamespace(requested=False)) == result


def test_checkpoint_pause_requeues_only_synonym_job(tmp_path, monkeypatch):
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
    from scripts import report_research, research_workflow

    path, _ = submission(tmp_path, monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "101")
    monkeypatch.setattr(research_workflow, "source_provenance", lambda: {})
    monkeypatch.setattr(report_research, "update_report", lambda: None)
    monkeypatch.setattr(job, "experiment", lambda *args: (_ for _ in ()).throw(CheckpointPause("saved")))
    calls = []
    monkeypatch.setattr(job.subprocess, "run", lambda command, **kwargs: calls.append(command))
    job.run(path)
    assert calls == [["scontrol", "requeue", "101"]]
    assert json.loads((path.parent / "receipt.json").read_text())["status"] == "paused"


def test_completed_report_publishes_paired_metrics_and_plot_exports(tmp_path, monkeypatch):
    from scripts import report_research

    monkeypatch.setattr(report_research, "ROOT", tmp_path)
    monkeypatch.setattr(report_research, "OUTPUT", tmp_path / "results")
    directory = tmp_path / "results/defender_comparison/synonym_robustness/fixture"
    job.write_record(directory / "manifest.json", {"job_id": "101", "held": False})
    job.write_record(directory / "receipt.json", {"status": "completed"})
    record = {"detector": "roberta", **paired_metrics([0.9, 0.2], [0.1, 0.7], [0.1, 0.1], [True, True])}
    job.write_record(directory / "results.json", {"records": [record], "num_pairs": 2,
        "num_changed": 2, "num_edits": 2, "actual_fraction_of_eligible_words": 0.05,
        "mean_semantic_original_vs_rewritten": 0.99, "mean_semantic_changed_only": 0.99,
        "num_rejected_proposals": 0})
    with (directory / "paired_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["detector", "score_delta"])
        writer.writeheader()
        writer.writerows([{"detector": "roberta", "score_delta": -0.8},
                          {"detector": "roberta", "score_delta": 0.5}])
    text = "\n".join(report_research.synonym_robustness(tmp_path / "docs/report.md"))
    assert "does not by itself establish" in text and "unchanged by construction" in text
    assert "changed_only_mean_score_delta" in text and "detected_to_undetected" in text
    for name in ("detection", "score-deltas"):
        for extension in ("png", "svg", "csv"):
            assert (tmp_path / f"docs/figures/research/synonyms-fixture-{name}.{extension}").stat().st_size > 0
