"""Review 10% and 25% synonym edits before frozen Gemma/RoBERTa scoring."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess

from scripts.research_chain import ROOT, OUTPUT, write_record
from scripts.synonym_robustness import MODEL, REVISION, inputs, release, variants, measure_variants

CONDITIONS = {"p10": 0.10, "p25": 0.25}
DETECTORS = ["roberta", "decoder_classifier"]


def submit():
    identity, plan = inputs()
    races = json.loads((OUTPUT / "architecture_races.json").read_text())
    path = OUTPUT / "defender_comparison/synonym_intensity" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") / "manifest.json"
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "inputs": identity, "baseline_plan": plan,
                "sample_indices": list(range(100)), "conditions": CONDITIONS, "detector_names": DETECTORS,
                "seed": 42, "model": MODEL, "revision": REVISION, "max_new_tokens": 1024, "target_budget": True,
                "detector_feedback": False, "training_updates": 0, "include_ensemble": False,
                "review_required": True, "max_restarts": 100, "held": True, "score_held": True,
                "capacity_race": races["jobs"]["roberta"]["job_id"], "jobs": {}}
    write_record(path, manifest)
    try:
        for stage in ("prepare", "score"):
            command = ["sbatch", "--parsable", "--hold", "--qos=qos_gpu_h100-dev", "--time=02:00:00",
                       "--signal=USR1@600", "--requeue", "--cpus-per-task=32", f"--job-name=synonym-{stage}-100",
                       f"--output={path.parent}/{stage}-%j.log"]
            if stage == "score":
                command += [f"--dependency=afterok:{manifest['jobs']['prepare']}", "--kill-on-invalid-dep=yes"]
            command += [str(ROOT / "scripts/run_research_slurm.sh"), "--synonym-intensity", str(path), stage]
            job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";", 1)[0]
            if not job.isdigit():
                raise RuntimeError(f"Unexpected Slurm response: {job!r}")
            manifest["jobs"][stage] = job
            manifest["job_id"] = manifest["jobs"]["prepare"]
            if stage == "score":
                manifest["capacity_blocker"] = job
            write_record(path, manifest)
            print(f"{stage}: {job}", flush=True)
    except BaseException:
        if manifest["jobs"]:
            subprocess.run(["scancel", *manifest["jobs"].values()], check=True)
        raise
    print(path, flush=True)
    return path


def selected_inputs(manifest):
    from ai_content_detector.rl_evasion.defender_comparison import workflow

    identity, plan = inputs()
    if identity != manifest["inputs"] or plan != manifest["baseline_plan"]:
        raise ValueError("Intensity inputs or warmed detectors changed since submission")
    if (manifest["conditions"] != CONDITIONS or manifest["sample_indices"] != list(range(100))
            or manifest["detector_names"] != DETECTORS or manifest["detector_feedback"]
            or manifest["training_updates"] != 0 or manifest["include_ensemble"] or not manifest["review_required"]):
        raise ValueError("Intensity experiment differs from the agreed independent 100-row protocol")
    base = json.loads((workflow.OUTPUT / "base_pairs.json").read_text())["evaluation"]
    baseline = json.loads((workflow.OUTPUT / "warmup/measurements.json").read_text())
    if len(base) != 500 or baseline["eligible_ids"] != [row["sample_id"] for row in base]:
        raise ValueError("Saved base-Qwen examples and baseline scores differ")
    pairs = [base[index] for index in manifest["sample_indices"]]
    if not all(row["eligible"] for row in pairs):
        raise ValueError("Selected base-Qwen pairs must be eligible")
    return pairs, baseline, plan


def prepare(manifest, directory, stop):
    pairs, _, _ = selected_inputs(manifest)
    for condition, rate in CONDITIONS.items():
        variants({**manifest, "rate": rate}, directory / condition, pairs, stop)
    write_record(directory / "preparation.json", {"status": "awaiting_coherence_review",
                 "rows_per_condition": len(pairs), "detector_scoring_started": False})


def review_proposals(directory, condition):
    from ai_content_detector.rl_evasion.benchmarking.synonym_robustness import apply_proposal

    raw = json.loads((directory / condition / "variants.json").read_text())
    records = []
    for row in raw:
        if any(item.get("reason") == "invalid_json" for item in row["rejections"]):
            applied = apply_proposal(row["original"], row, row["response"], recover_complete=True)
            if applied.get("json_recovery"):
                row = {**row, "raw_parse_edits": row["edits"], "raw_parse_rejections": row["rejections"], **applied}
        records.append(row)
    return records


def reviewed_variants(manifest, directory):
    from ai_content_detector.rl_evasion.benchmarking.synonym_robustness import apply_proposal
    from ai_content_detector.rl_evasion.defender_comparison.protocol import digest

    review = json.loads((directory / "review.json").read_text())
    if (not review.get("review_before_new_scoring") or not review.get("reviewer")
            or set(review.get("conditions", {})) != set(CONDITIONS)):
        raise ValueError("Both edit conditions require an identified review before new scoring")
    result = {}
    for condition, rate in CONDITIONS.items():
        raw = review_proposals(directory, condition)
        decisions = review["conditions"][condition]
        if (len(raw) != 100 or len(decisions) != 100
                or review["source_variants_sha256"][condition] != digest(raw)):
            raise ValueError("Coherence review is incomplete or refers to different proposals")
        records = []
        for index, (row, decision) in enumerate(zip(raw, decisions)):
            if row["index"] != index or decision["index"] != index or not decision.get("coherent_after_review"):
                raise ValueError("Every row must have a coherence decision in the original order")
            edits = {edit["index"]: edit for edit in row["edits"]}
            accepted = decision["accepted"]
            rejected = [edit["index"] for edit in decision["rejected"]]
            corrections = decision.get("corrections", [])
            corrected = [edit["index"] for edit in corrections]
            if (len(accepted) != len(set(accepted)) or len(rejected) != len(set(rejected))
                    or set(accepted) & set(rejected) or set(accepted) | set(rejected) != set(edits)
                    or any(not edit.get("reason") for edit in decision["rejected"])
                    or len(corrected) != len(set(corrected)) or not set(corrected) <= set(rejected)
                    or any(not edit.get("reason") for edit in corrections)
                    or len(accepted) + len(corrected) > math.floor(rate * row["eligible_content_words"])):
                raise ValueError("Every proposed edit requires one accepted or reasoned rejection decision")
            response = json.dumps({"edits": [{"index": token, "replacement": edits[token]["replacement"].lower()}
                                             for token in accepted] + [
                                                 {"index": edit["index"], "replacement": edit["replacement"]}
                                                 for edit in corrections]})
            applied = apply_proposal(row["original"], row, response)
            if applied["rejections"] or len(applied["edits"]) != len(accepted) + len(corrected):
                raise ValueError("Reviewed edits no longer match the permitted source spans and synonyms")
            records.append({**row, "proposed_text": row["text"], "proposed_edits": row["edits"],
                            **applied, "coherence_review": decision,
                            "proposal_rejections": row["rejections"], "reviewer": review["reviewer"]})
        result[condition] = records
    return result


def freeze_review(path):
    from ai_content_detector.rl_evasion.defender_comparison.protocol import digest

    path = Path(path)
    manifest = json.loads(path.read_text())
    if json.loads((path.parent / "prepare_receipt.json").read_text())["status"] != "completed":
        raise ValueError("Preparation must finish before coherence review can be frozen")
    records = reviewed_variants(manifest, path.parent)
    hashes = {condition: digest(rows) for condition, rows in records.items()}
    receipt_path = path.parent / "review_receipt.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text())["reviewed_sha256"] != hashes:
        raise ValueError("The already frozen coherence review cannot be changed")
    for condition, rows in records.items():
        write_record(path.parent / condition / "reviewed_variants.json", rows)
    receipt = {"reviewed_sha256": hashes,
               "review_sha256": digest(json.loads((path.parent / "review.json").read_text())),
               "review_before_new_scoring": True, "frozen_at": datetime.now(timezone.utc).isoformat()}
    write_record(receipt_path, receipt)
    return records


def release_score(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if not manifest["score_held"]:
        raise ValueError("Intensity score job is already released")
    freeze_review(path)
    subprocess.run(["scontrol", "release", manifest["jobs"]["score"]], check=True)
    manifest.update(score_held=False, scoring_released_at=datetime.now(timezone.utc).isoformat())
    write_record(path, manifest)


def score(manifest, directory, stop):
    from ai_content_detector.rl_evasion.defender_comparison.protocol import digest

    pairs, baseline, plan = selected_inputs(manifest)
    frozen = json.loads((directory / "review_receipt.json").read_text())
    reviewed = reviewed_variants(manifest, directory)
    if (frozen["review_sha256"] != digest(json.loads((directory / "review.json").read_text()))
            or frozen["reviewed_sha256"] != {condition: digest(rows) for condition, rows in reviewed.items()}):
        raise ValueError("The coherence review changed after it was frozen")
    results = []
    for condition, rate in CONDITIONS.items():
        rows = json.loads((directory / condition / "reviewed_variants.json").read_text())
        if digest(rows) != frozen["reviewed_sha256"][condition]:
            raise ValueError("Scored variants differ from the frozen coherence review")
        result = measure_variants({**manifest, "rate": rate}, directory / condition, pairs, baseline, plan, rows, stop)
        result.update(num_review_corrections=sum(len(row["coherence_review"].get("corrections", [])) for row in rows),
                      num_proposed_edits=sum(len(row["proposed_edits"]) for row in rows),
                      num_coherence_rejections=sum(len(row["coherence_review"]["rejected"]) for row in rows))
        results.append({"condition": condition, "target_rate": rate, **result})
    write_record(directory / "results.json", {"conditions": results, "review_before_new_scoring": True,
                 "reviewed_sha256": frozen["reviewed_sha256"], "completed_at": datetime.now(timezone.utc).isoformat()})


def run(path, stage):
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
    from scripts.research_workflow import source_provenance
    from scripts.report_research import update_report

    path = Path(path)
    manifest = json.loads(path.read_text())
    job = os.environ["SLURM_JOB_ID"]
    if job != manifest["jobs"][stage]:
        raise ValueError("Intensity stage does not match its submitted Slurm job")
    with (path.parent / f"{stage}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = path.parent / f"{stage}_receipt.json"
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        if previous.get("status") == "completed":
            return
        receipt = {"status": "running", "attempt": previous.get("attempt", 0) + 1,
                   "slurm_job_id": job, "started_at": datetime.now(timezone.utc).isoformat()}
        write_record(receipt_path, receipt)
        try:
            with StopRequest() as stop:
                receipt.update(source_provenance())
                write_record(receipt_path, receipt)
                (prepare if stage == "prepare" else score)(manifest, path.parent, stop)
            receipt["status"] = "completed"
        except CheckpointPause as error:
            receipt.update(status="paused", error=str(error))
        except BaseException as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_record(receipt_path, receipt)
            write_record(path.parent / f"{stage}-attempt-{receipt['attempt']}.json", receipt)
            update_report()
        if receipt["status"] == "paused":
            if receipt["attempt"] >= manifest["max_restarts"]:
                raise RuntimeError("Intensity stage reached its restart limit")
            subprocess.run(["scontrol", "requeue", job], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("submit")
    for mode in ("release-prepare", "release-score"):
        modes.add_parser(mode).add_argument("manifest", type=Path)
    execution = modes.add_parser("run")
    execution.add_argument("manifest", type=Path)
    execution.add_argument("stage", choices=["prepare", "score"])
    arguments = parser.parse_args()
    if arguments.mode == "submit":
        submit()
    elif arguments.mode == "release-prepare":
        release(arguments.manifest)
    elif arguments.mode == "release-score":
        release_score(arguments.manifest)
    else:
        run(arguments.manifest, arguments.stage)
