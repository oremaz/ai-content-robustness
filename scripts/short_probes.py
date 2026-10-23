"""Run independent ten-update attackers against the three warmed defenders."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess

from scripts.architecture_races import DEFENDERS, inputs, save_metrics, write_metrics
from scripts.research_chain import ROOT, OUTPUT, write_record


def split_audit():
    root = OUTPUT / "defender_comparison"
    partition = json.loads((root / "prompt_manifest.json").read_text())
    pairs = json.loads((root / "base_pairs.json").read_text())
    evaluation = [row["prompt"] for row in pairs["evaluation"]]
    training = set(partition["train"])
    warmup = {row["prompt"] for row in pairs["warmup"]}
    if len(evaluation) != 500:
        raise ValueError("Probe evaluation requires the 500 recorded evaluation rows")
    if set(evaluation) & (training | warmup) or not set(evaluation) <= set(partition["evaluation"]):
        raise ValueError("Probe evaluation overlaps fitting data or differs from the reserved partition")
    return {"training_rows": len(partition["train"]), "unique_training_prompts": len(training),
            "reserved_prompts": len(partition["evaluation"]),
            "scored_prompts": len(evaluation), "exact_prompt_overlap_with_fitting": 0,
            "unique_scored_prompts": len(set(evaluation)),
            "usage": "Held-out diagnostic set already used for budget selection"}


def submit():
    identity, plan = inputs()
    races = json.loads((OUTPUT / "architecture_races.json").read_text())
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = OUTPUT / "defender_comparison/probes_10" / identifier / "manifest.json"
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "inputs": identity,
                "baseline_plan": plan, "split_audit": split_audit(), "attacker_steps": 10, "defenders": DEFENDERS,
                "initialization": "base_Qwen", "defender_updates": 0,
                "evaluation": "existing_500_reserved_prompts_used_for_budget_selection",
                "qos": "qos_gpu_h100-dev", "time_limit": "02:00:00", "gpu_limit": 3,
                "max_restarts": 100, "held": True, "metrics_file": str(path.parent / "metrics.csv"),
                "reference": str(OUTPUT / "defender_comparison/checkpoint_evaluations/ensemble/step-502/results.json"),
                "comparison": "Unmatched training targets and learning-rate horizons; descriptive only",
                "race_jobs": {name: row["job_id"] for name, row in races["jobs"].items()}, "jobs": {}}
    write_record(path, manifest)
    write_metrics(Path(manifest["metrics_file"]), [])
    try:
        for name in DEFENDERS:
            directory = path.parent / name
            directory.mkdir(parents=True, exist_ok=True)
            command = ["sbatch", "--parsable", "--hold", "--qos=qos_gpu_h100-dev", "--time=02:00:00",
                       "--signal=USR1@600", "--requeue", "--cpus-per-task=32", f"--job-name=probe10-{name}",
                       f"--output={directory}/slurm-%j.log", str(ROOT / "scripts/run_research_slurm.sh"),
                       "--short-probe", str(path), name]
            job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";", 1)[0]
            if not job.isdigit():
                raise RuntimeError(f"Unexpected Slurm response: {job!r}")
            manifest["jobs"][name] = {"job_id": job, "directory": str(directory), "command": command}
            write_record(path, manifest)
            print(f"{name}: {job}", flush=True)
    except BaseException:
        if manifest["jobs"]:
            subprocess.run(["scancel", *[row["job_id"] for row in manifest["jobs"].values()]], check=True)
        raise
    return path


def release(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if not manifest["held"] or set(manifest["jobs"]) != set(DEFENDERS):
        raise ValueError("All three probes must be held before release")
    race_ids = ",".join(manifest["race_jobs"].values())

    def states():
        text = subprocess.check_output(["squeue", "--noheader", "--jobs", race_ids,
                                        "--format=%i|%T"], text=True)
        return dict(line.strip().split("|") for line in text.splitlines() if line.strip())

    expected = {job: "PENDING" for job in manifest["race_jobs"].values()}
    if states() != expected:
        raise RuntimeError("A race is no longer pending; inspect GPU capacity before releasing probes")
    subprocess.run(["scontrol", "hold", race_ids], check=True)
    try:
        if states() != expected:
            raise RuntimeError("A race started during capacity handover; probes remain held")
        for name, race in manifest["race_jobs"].items():
            dependency = f"afterany:{manifest['jobs'][name]['job_id']}"
            subprocess.run(["scontrol", "update", f"JobId={race}", f"Dependency={dependency}"], check=True)
        manifest["capacity_dependencies"] = {name: f"afterany:{row['job_id']}"
                                              for name, row in manifest["jobs"].items()}
        write_record(path, manifest)
        subprocess.run(["scontrol", "release", ",".join(row["job_id"] for row in manifest["jobs"].values())], check=True)
        manifest.update(held=False, released_at=datetime.now(timezone.utc).isoformat())
        write_record(path, manifest)
    finally:
        subprocess.run(["scontrol", "release", race_ids], check=True)


def probe(manifest, name, stop):
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
    from ai_content_detector.rl_evasion.defender_comparison import workflow
    from ai_content_detector.rl_evasion.defender_comparison.readiness import require_readiness

    identity, plan = inputs()
    if identity != manifest["inputs"] or plan != manifest["baseline_plan"]:
        raise ValueError("Probe inputs or warmed defenders changed since submission")
    if manifest["attacker_steps"] != 10 or manifest["defender_updates"] != 0 or manifest["defenders"] != DEFENDERS:
        raise ValueError("Probe protocol differs from the agreed ten-update frozen-defender experiment")
    directory = Path(manifest["jobs"][name]["directory"])
    result_path = directory / "results.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    require_readiness(workflow.OUTPUT, [DEFENDERS[name]])
    attacker, pool = workflow.attacker_setup(plan, name, directory / "attacker", names=[DEFENDERS[name]])
    try:
        attacker.config.num_train_epochs = 1
        if stop.requested:
            raise CheckpointPause("Probe model setup paused")
        attacker.train(max_steps=10)
        write_record(directory / "training_history.json", attacker.training_history)
        candidates = [p for p in (directory / "attacker").glob("checkpoint-*")
                      if (p / "restart_complete.pt").exists()]
        checkpoint = max(candidates, key=lambda p: int(p.name.split("-")[-1]))
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if state["global_step"] != 10 or state["max_steps"] != 10:
            raise ValueError("Probe did not finish exactly ten optimizer updates")
        if stop.requested:
            raise CheckpointPause("Ten-update training checkpoint retained before evaluation")
        result = workflow.evaluate_attacker(attacker, pool, directory / "evaluation", f"probe10/{name}")
        result.update(attacker_steps=10, checkpoint=str(checkpoint), config=asdict(attacker.config),
                      training_targets=list(pool.specifications), completed_at=datetime.now(timezone.utc).isoformat())
        write_record(result_path, result)
        return result
    finally:
        pool.clear()


def run(path, name):
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
    from scripts.research_workflow import source_provenance
    from scripts.report_research import update_report

    manifest = json.loads(Path(path).read_text())
    directory = Path(manifest["jobs"][name]["directory"])
    job_id = os.environ["SLURM_JOB_ID"]
    if manifest["jobs"][name]["job_id"] != job_id:
        raise ValueError("Probe does not match its submitted Slurm job")
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = directory / "receipt.json"
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        if previous.get("status") == "completed":
            return
        receipt = {"status": "running", "slurm_job_id": job_id, "attempt": previous.get("attempt", 0) + 1,
                   "started_at": datetime.now(timezone.utc).isoformat()}
        write_record(receipt_path, receipt)
        try:
            with StopRequest() as stop:
                receipt.update(source_provenance())
                write_record(receipt_path, receipt)
                result = probe(manifest, name, stop)
                save_metrics(manifest["metrics_file"], name, 1, "after_attack", result, 10)
            receipt["status"] = "completed"
        except CheckpointPause as error:
            receipt.update(status="paused", error=str(error))
        except BaseException as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_record(receipt_path, receipt)
            write_record(directory / f"attempt-{receipt['attempt']}.json", receipt)
            update_report()
        if receipt["status"] == "paused":
            if receipt["attempt"] >= manifest["max_restarts"]:
                raise RuntimeError("Short probe reached its restart limit")
            subprocess.run(["scontrol", "requeue", job_id], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("submit")
    handover = modes.add_parser("release")
    handover.add_argument("manifest", type=Path)
    execution = modes.add_parser("run")
    execution.add_argument("manifest", type=Path)
    execution.add_argument("defender", choices=DEFENDERS)
    arguments = parser.parse_args()
    if arguments.mode == "submit":
        print(submit())
    elif arguments.mode == "release":
        release(arguments.manifest)
    else:
        run(arguments.manifest, arguments.defender)
