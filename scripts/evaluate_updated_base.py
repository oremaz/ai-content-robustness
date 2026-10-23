"""Evaluate updated defenders on the original reserved base-Qwen outputs."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess

from scripts import architecture_races
from scripts.research_chain import write_record
from scripts.retained_races import cached_measurement, sha256


def prepare(source_path):
    from ai_content_detector.rl_evasion.defender_comparison import workflow

    source_path = Path(source_path).resolve()
    source = json.loads(source_path.read_text())
    path = source_path.parent / "base_retest/manifest.json"
    if path.exists():
        raise ValueError("A base-model evaluation is already prepared for this defender update")
    root = workflow.OUTPUT
    inputs = {name: sha256(root / name) for name in
              ("base_pairs.json", "prompt_manifest.json", "validation_thresholds.json")}
    if any(source["inputs"]["input_sha256"][name] != value for name, value in inputs.items()):
        raise ValueError("Base-model evaluation inputs differ from the defender experiment")
    base = json.loads((root / "base_pairs.json").read_text())["evaluation"]
    thresholds = json.loads((root / "validation_thresholds.json").read_text())
    baseline = json.loads((root / "warmup/measurements.json").read_text())
    if workflow.checkpoint_identity(workflow.warm_checkpoints()) != source["all_initial_defenders"]:
        raise ValueError("Original warmed defenders changed after their baseline evaluation")
    branches = {}
    for name, names in source["protocol"]["defenders"].items():
        directory = Path(source["jobs"][name]["directory"]).resolve()
        receipt = json.loads((directory / "receipt.json").read_text())
        if receipt["status"] != "completed" or receipt["completed_rounds"] != 1:
            raise ValueError("Every source defender update must be completed")
        checkpoints = {detector: str(directory / "round_1/defenders" / detector / "round_1") for detector in names}
        branches[name] = {"checkpoints": checkpoints, "identity": workflow.checkpoint_identity(checkpoints),
                          "before": cached_measurement(base, baseline, names, thresholds)}
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "source_manifest": str(source_path),
                "input_sha256": inputs, "baseline_plan": source["baseline_plan"], "branches": branches,
                "protocol": {"rounds": 1, "attacker_steps_per_round": 0,
                             "defenders": source["protocol"]["defenders"], "evaluation": "saved_base_Qwen",
                             "generated_texts": len(base), "threshold": 0.5},
                "qos": "qos_gpu_h100-dev", "time_limit": "02:00:00", "gpu_limit": 3,
                "cpus_by_branch": {"ensemble": 48}, "max_restarts": 100,
                "runner_mode": "--updated-base", "job_prefix": "base-retest",
                "metrics_file": str(path.parent / "metrics.csv"), "held": True, "jobs": {}}
    write_record(path, manifest)
    architecture_races.write_metrics(Path(manifest["metrics_file"]), [])
    source["base_evaluation_manifest"] = str(path)
    write_record(source_path, source)
    return path


def evaluate(manifest, name, directory):
    from ai_content_detector.rl_evasion.defender_comparison import workflow
    from ai_content_detector.rl_evasion.defender_comparison.protocol import evaluate_pool

    branch = manifest["branches"][name]
    for filename, expected in manifest["input_sha256"].items():
        if sha256(workflow.OUTPUT / filename) != expected:
            raise ValueError("Base-model evaluation dataset changed")
    if workflow.checkpoint_identity(branch["checkpoints"]) != branch["identity"]:
        raise ValueError("Updated defender changed after evaluation submission")
    result_path = directory / "evaluation.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    pairs = json.loads((workflow.OUTPUT / "base_pairs.json").read_text())["evaluation"]
    if len(pairs) != manifest["protocol"]["generated_texts"]:
        raise ValueError("Reserved base-model evaluation count changed")
    names = manifest["protocol"]["defenders"][name]
    pool = workflow.group_pool(manifest["baseline_plan"], name, directory / "defenders",
                               checkpoints=branch["checkpoints"], names=names)
    try:
        thresholds = json.loads((workflow.OUTPUT / "validation_thresholds.json").read_text())
        evaluate_pool(pool, pairs, directory / "scores", thresholds)
        artifact = json.loads((directory / "scores/measurements.json").read_text())
        result = {**cached_measurement(pairs, artifact, names, thresholds), "attacker_updates": 0,
                  "generator": "Qwen/Qwen3.5-9B-Base", "adaptation": "none", "defender_round": 1,
                  "completed_at": datetime.now(timezone.utc).isoformat()}
        write_record(result_path, result)
        return result
    finally:
        pool.clear()


def run(path, name):
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
    from scripts.report_research import update_report
    from scripts.research_workflow import source_provenance

    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    job = manifest["jobs"][name]
    if job["job_id"] != os.environ["SLURM_JOB_ID"]:
        raise ValueError("Base-model evaluator does not match the submitted Slurm job")
    directory = Path(job["directory"])
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous_path = directory / "receipt.json"
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
        receipt = {"status": "running", "attempt": previous.get("attempt", 0) + 1,
                   "slurm_job_id": job["job_id"], "started_at": datetime.now(timezone.utc).isoformat()}
        write_record(previous_path, receipt)
        try:
            with StopRequest():
                receipt.update(source_provenance())
                before = {**manifest["branches"][name]["before"], "attacker_updates": 0}
                architecture_races.save_metrics(manifest["metrics_file"], name, 1, "before", before, 0)
                result = evaluate(manifest, name, directory)
                architecture_races.save_metrics(manifest["metrics_file"], name, 1, "after_defense", result, 0)
            receipt["status"] = "completed"
        except CheckpointPause as error:
            receipt.update(status="paused", error=str(error))
        except BaseException as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_record(previous_path, receipt)
            update_report()
        if receipt["status"] == "paused":
            if receipt["attempt"] >= manifest["max_restarts"]:
                raise RuntimeError("Base-model evaluation reached its restart limit")
            subprocess.run(["scontrol", "requeue", job["job_id"]], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    preparing = commands.add_parser("prepare"); preparing.add_argument("source", type=Path)
    submission = commands.add_parser("submit"); submission.add_argument("manifest", type=Path)
    releasing = commands.add_parser("release"); releasing.add_argument("manifest", type=Path)
    running = commands.add_parser("run"); running.add_argument("manifest", type=Path); running.add_argument("name")
    args = parser.parse_args()
    if args.mode == "prepare":
        print(prepare(args.source))
    elif args.mode == "run":
        run(args.manifest, args.name)
    else:
        path = args.manifest.resolve()
        manifest = json.loads(path.read_text())
        if args.mode == "submit":
            if manifest["jobs"]:
                raise ValueError("Base-model evaluation jobs already exist")
            architecture_races.submit_jobs(path, manifest)
        else:
            if not manifest["held"] or len(manifest["jobs"]) != 3:
                raise ValueError("Expected all three evaluation jobs to be held")
            subprocess.run(["scontrol", "release", ",".join(job["job_id"] for job in manifest["jobs"].values())], check=True)
            manifest.update(held=False, released_at=datetime.now(timezone.utc).isoformat())
            write_record(path, manifest)
