"""Owner-only scoring after blind text attack selection is frozen."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(path):
    import importlib.metadata
    from unittest.mock import patch
    from watermark_survival_pilot import dependency_map, save
    from ai_content_detector.watermarking.instruct_text import textseal2_config, textseal2_score
    from ai_content_detector.watermarking.preservation import joint_outcomes
    from ai_content_detector.watermarking.exposure_support import evidence_lower_bound
    config = json.loads(Path(path).read_text())
    root, generated = Path(config["directory"]), Path(config["generated"])
    completion = json.loads((root / "quality_complete.json").read_text())
    if completion["signature"] != config["signature"] or sha(root / "selection.json") != completion["selection_sha256"]:
        raise ValueError("Quality-based selection changed before owner scoring")
    for filename, expected in config["input_hashes"].items():
        if sha(filename) != expected:
            raise ValueError("Owner-scoring input changed: " + filename)
    selection = json.loads((root / "selection.json").read_text())
    if selection["owner_scores_used"] is not False:
        raise ValueError("Owner feedback entered blind selection")
    sys.path.insert(0, config["textseal2_source"])
    with patch("importlib.metadata.packages_distributions", return_value=dependency_map()):
        from transformers import AutoTokenizer
        from textseal.watermarking.detector import build_detector
    tokenizer = AutoTokenizer.from_pretrained(config["generator"]["path"], local_files_only=True)
    detector = build_detector(tokenizer, textseal2_config(config["watermark"]["secret_key"]))
    originals = json.loads((generated / "originals.json").read_text())
    original_scores = {r["id"]: textseal2_score(detector, r["text"]) for r in originals}
    original_decisions = {key: bool(value["p_value"] <= .001) for key, value in original_scores.items()}
    summaries, all_scores, certificates = {}, {}, {}
    for arm, choices in selection["selected"].items():
        quality_path = root / (arm + "-quality.json")
        if sha(quality_path) != completion["quality_hashes"][arm]:
            raise ValueError("Frozen quality measurements changed")
        quality = {(r["id"], r["attempt"]): r for r in json.loads(quality_path.read_text())}
        attempts = {(r["id"], r["attempt"]): r for r in json.loads((generated / (arm + ".json")).read_text())}
        scores = {key: textseal2_score(detector, row["text"]) for key, row in attempts.items()}
        if config.get("exposure_guard"):
            certificates[arm]=[]
            for key,row in attempts.items():
                bound=evidence_lower_bound(row["tokens"],row["evidence_floors"],row["guard_support_membership"])
                generated_scores=detector.get_scores_by_t([row["tokens"]],scoring_method="v2",seen_windows=set())[0]
                if len(generated_scores)!=bound["scored_pairs"] or sum(generated_scores)+1.e-9<bound["score_lower_bound"]:
                    raise ValueError("Observed native evidence violates its frozen support-path bound")
                roundtrip=tokenizer.encode(row["text"],add_special_tokens=False)==row["tokens"]
                p_bound=detector.get_pvalue(bound["score_lower_bound"],bound["scored_pairs"])
                if roundtrip and scores[key]["p_value"]>p_bound+1.e-12:
                    raise ValueError("Retokenized text violates the certified Gamma tail bound")
                certificates[arm].append({"id":key[0],"attempt":key[1],**bound,
                    "tokenization_roundtrip":roundtrip,"native_generated_p_value_upper_bound":p_bound,
                    "text_detection_certified":bool(roundtrip and p_bound<=.001)})
        selected_decisions, preservation = {}, {}
        for choice in choices:
            key = choice["id"], choice["attempt"]
            row, outcome = attempts[key], quality[key]
            if hashlib.sha256(row["text"].encode()).hexdigest() != choice["candidate_sha256"]:
                raise ValueError("Selected text changed")
            if choice["passed"] != outcome["passed"]:
                raise ValueError("Selection and preservation gate disagree")
            selected_decisions[key[0]] = bool(scores[key]["p_value"] <= .001)
            preservation[key[0]] = {"passed": outcome["passed"], "failures": outcome["failures"]}
        summaries[arm] = joint_outcomes(original_decisions, selected_decisions, preservation)
        all_scores[arm] = [{"id": key[0], "attempt": key[1], **value} for key, value in scores.items()]
    null_scores = [textseal2_score(detector, row["human_text"]) for row in config["data"]["evaluation"]]
    result = {"quality_manifest": str(Path(path).resolve()), "quality_manifest_sha256": sha(path),
              "quality_completion_sha256": sha(root / "quality_complete.json"),
              "selection_sha256": sha(root / "selection.json"), "scorer_source_sha256": sha(__file__),
              "n": len(originals), "original_detected": sum(original_decisions.values()),
              "nominal_p_value_threshold": .001, "conditions": summaries,
              "original_scores": original_scores, "attempt_scores": all_scores,
              "support_path_certificates":certificates,
              "human_null": {"n": len(null_scores), "detected": sum(bool(r["p_value"] <= .001) for r in null_scores),
                             "scores": null_scores,
                             "limitation": f"{len(null_scores)} human passages cannot precisely validate a 0.1 percent fixed-key FPR. These are diagnostic counts at the unchanged nominal Gamma threshold, not a new calibration guarantee."},
              "scope": config["scope"],
              "attack_protocol": {"source_ngram_order": config["sampling"].get("source_ngram_order"),
                                  "rewrite_instruction": config["rewrite_instruction"],
                                  "selection": config["selection"]}}
    save(root / "owner_results.json", result)
    print(json.dumps({"original_detected": result["original_detected"], "conditions": summaries}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    run(parser.parse_args().manifest)
