"""Submit independent research branches with native Slurm dependencies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess

from scripts.submit_research import ROOT, OUTPUT


def write_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def branches(image_job):
    nodes = [("image_notebook", [("afterany", str(image_job))]),
             ("comparison_baseline", []),
             ("comparison_warmup", [("afterok", "comparison_baseline")])]
    for group in ("encoder", "decoder", "ensemble"):
        dependencies = [("afterok", "comparison_warmup")]
        if group == "ensemble":
            dependencies.append(("afterany", "image_notebook"))
        nodes.append((f"comparison_rl_{group}", dependencies))
    nodes.extend((f"comparison_race_{group}", [("afterok", f"comparison_rl_{group}")])
                 for group in ("encoder", "decoder", "ensemble"))
    return nodes


def stage_resources(stage):
    if stage == "image_notebook":
        return {"cpus": 96, "walltime": "20:00:00"}
    if stage == "comparison_warmup":
        return {"cpus": 96, "walltime": "04:00:00"}
    if stage.endswith("_ensemble"):
        return {"cpus": 48, "walltime": "20:00:00"}
    return {"cpus": 32, "walltime": "04:00:00" if stage in {
        "comparison_baseline", "comparison_warmup"} else "20:00:00"}


def submit(image_job, *, qos="qos_gpu_h100-t3", walltime=None, hold=True):
    receipt = json.loads((OUTPUT / "text_notebook.json").read_text())
    if receipt.get("status") != "completed":
        raise RuntimeError("The full text notebook must complete before submitting this comparison")
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = OUTPUT / "chains" / identifier / "manifest.json"
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "adopted_image_job": str(image_job),
                "qos": qos, "walltime": walltime or "per stage", "gpu_limit": 3, "max_restarts": 100,
                "held": hold, "jobs": {}}
    write_record(path, manifest)
    try:
        for stage, dependencies in branches(image_job):
            resources = stage_resources(stage)
            command = ["sbatch", "--parsable", f"--qos={qos}", f"--time={walltime or resources['walltime']}",
                       "--signal=USR1@600", "--requeue", "--kill-on-invalid-dep=yes",
                       f"--cpus-per-task={resources['cpus']}", f"--job-name=research-{stage}",
                       f"--output={OUTPUT}/slurm-%j-{stage}.log"]
            if hold:
                command.append("--hold")
            resolved = [(kind, manifest["jobs"][name]["job_id"] if name in manifest["jobs"] else name)
                        for kind, name in dependencies]
            if resolved:
                command.append("--dependency=" + ",".join(f"{kind}:{job}" for kind, job in resolved))
            command += [str(ROOT / "scripts/run_research_slurm.sh"), "--chain", str(path), stage]
            result = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";", 1)[0]
            if not result.isdigit():
                raise RuntimeError(f"Unexpected sbatch response for {stage}: {result!r}")
            manifest["jobs"][stage] = {"job_id": result, "dependencies": resolved, "command": command}
            write_record(path, manifest)
            print(f"{stage}: {result}", flush=True)
    except BaseException:
        jobs = [job["job_id"] for job in manifest["jobs"].values()]
        if jobs:
            subprocess.run(["scancel", *jobs], check=True)
        raise
    return path


def run(path, stage):
    manifest = json.loads(Path(path).read_text())
    job_id = os.environ["SLURM_JOB_ID"]
    if manifest["jobs"][stage]["job_id"] != job_id:
        raise ValueError("The stage does not match this Slurm chain job")
    with (OUTPUT / f"{stage}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = OUTPUT / f"{stage}.json"
        if receipt_path.exists() and json.loads(receipt_path.read_text()).get("status") == "completed":
            print(f"{stage}: already completed; retaining its original execution receipt", flush=True)
            return
        submission = {"stage": stage, "job_id": job_id, "chain_manifest": str(path),
                      "restart": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
                      "submitted_at": datetime.now(timezone.utc).isoformat()}
        with (OUTPUT / "submissions.jsonl").open("a") as stream:
            stream.write(json.dumps(submission) + "\n")
        from scripts.research_workflow import run_stage
        from ai_content_detector.rl_evasion.checkpointing import StopRequest
        try:
            with StopRequest():
                run_stage(stage)
        except SystemExit as error:
            if error.code != 75:
                raise
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "paused" or receipt.get("slurm_job_id") != job_id:
                raise RuntimeError("Refusing requeue without this job's saved pause receipt") from error
            if submission["restart"] >= manifest["max_restarts"]:
                raise RuntimeError("Research chain restart limit reached") from error
            write_record(Path(path).parent / f"{job_id}-restart-{submission['restart']}.json",
                         {**submission, "outcome": "checkpointed; requeue requested"})
            print(f"{stage}: checkpoint saved; requeueing job {job_id}", flush=True)
            subprocess.run(["scontrol", "requeue", job_id], check=True)
        else:
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "completed" or receipt.get("slurm_job_id") != job_id:
                raise RuntimeError("Stage returned without its completed execution receipt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    submission = modes.add_parser("submit")
    submission.add_argument("--after-image-job", type=int, required=True)
    submission.add_argument("--qos", default="qos_gpu_h100-t3")
    submission.add_argument("--time", help="Override the per-stage allocation limits")
    execution = modes.add_parser("run")
    execution.add_argument("manifest", type=Path)
    execution.add_argument("stage")
    args = parser.parse_args()
    if args.mode == "submit":
        print(submit(args.after_image_job, qos=args.qos, walltime=args.time))
    else:
        run(args.manifest, args.stage)


if __name__ == "__main__":
    main()
