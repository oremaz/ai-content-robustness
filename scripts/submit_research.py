"""Submit one resumable research stage at a time on Jean Zay."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from ai_content_detector.rl_evasion.defender_comparison.plan import STAGES as COMPARISON_STAGES

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/research_workflow"
STAGES = ["text_notebook", "image_notebook", "grpo_text", "multispin",
          "ddpo_image", "benchmark", "arms_race", "meta_adapt", *COMPARISON_STAGES]


def wait_for_existing_job(job_id):
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "BOOT_FAIL",
                "DEADLINE", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED"}
    print(f"Waiting for existing Slurm job {job_id} before checking completed stages.", flush=True)
    while True:
        output = subprocess.check_output(["sacct", "-n", "-X", "-j", str(job_id),
                                          "--format=State%30", "--parsable2"], text=True)
        states = [line.split("|", 1)[0].strip().split()[0].rstrip("+")
                  for line in output.splitlines() if line.strip()]
        if not states:
            raise RuntimeError(f"No Slurm accounting entry for adopted job {job_id}")
        if all(state in terminal for state in states):
            print(f"Existing job {job_id} finished: {states}", flush=True)
            return
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="+", choices=STAGES)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--after-job", type=int, help="Wait for this existing Slurm job before the first stage")
    parser.add_argument("--wait-for-controller", action="store_true", help="Wait for the current sequential controller to finish")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "controller_requests.jsonl").open("a") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "stages": args.stages,
                                 "requested_at": datetime.now(timezone.utc).isoformat(),
                                 "wait_for_controller": args.wait_for_controller}) + "\n")
    if args.wait_for_controller:
        print("Waiting for the active controller; comparison GPU jobs start after notebook completion.", flush=True)
    with (OUTPUT / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_controller else fcntl.LOCK_NB))
        if args.after_job is not None:
            wait_for_existing_job(args.after_job)
            args.after_job = None
        for stage in args.stages:
            if stage in COMPARISON_STAGES:
                for notebook in ("text_notebook",):
                    receipt = OUTPUT / f"{notebook}.json"
                    if not receipt.exists() or json.loads(receipt.read_text()).get("status") != "completed":
                        raise RuntimeError(f"{notebook} did not complete; no comparison GPU job will be submitted")
                if stage.startswith(("comparison_rl_", "comparison_race_")):
                    from ai_content_detector.rl_evasion.defender_comparison.readiness import require_readiness
                    from ai_content_detector.rl_evasion.defender_comparison.plan import GROUPS
                    require_readiness(OUTPUT / "defender_comparison", GROUPS[stage.rsplit("_", 1)[1]])
            status_path = OUTPUT / f"{stage}.json"
            if status_path.exists() and json.loads(status_path.read_text()).get("status") == "completed":
                print(f"{stage}: already completed", flush=True)
                continue
            for attempt in range(args.max_attempts):
                command = ["sbatch", "--parsable", "--wait",
                           f"--cpus-per-task={96 if stage in {'image_notebook', 'meta_adapt'} or stage in COMPARISON_STAGES else 24}",
                           f"--job-name=research-{stage}",
                           f"--output={OUTPUT}/slurm-%j-{stage}.log",
                           str(ROOT / "scripts/run_research_slurm.sh"), stage]
                if args.after_job is not None:
                    command.insert(3, f"--dependency=afterany:{args.after_job}")
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                           stderr=sys.stderr, text=True)
                job_id = process.stdout.readline().strip().split(";", 1)[0]
                if not job_id.isdigit():
                    process.wait()
                    raise RuntimeError(f"No Slurm job ID returned for {stage}: {job_id!r}")
                receipt = {"stage": stage, "job_id": job_id, "attempt": attempt + 1,
                           "submitted_at": datetime.now(timezone.utc).isoformat(), "command": command}
                with (OUTPUT / "submissions.jsonl").open("a") as stream:
                    stream.write(json.dumps(receipt) + "\n")
                print(f"{stage}: submitted job {job_id}", flush=True)
                return_code = process.wait()
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                if status.get("slurm_job_id") != job_id:
                    raise RuntimeError(f"Job {job_id} produced no stage receipt; inspect its Slurm log")
                print(f"{stage}: job {job_id} {status.get('status')} (exit {return_code})", flush=True)
                if status.get("status") == "completed" and return_code == 0:
                    break
                if status.get("status") != "paused":
                    raise RuntimeError(f"{stage} failed in job {job_id}; inspect {status_path}")
            else:
                raise RuntimeError(f"{stage} reached the {args.max_attempts}-attempt limit")


if __name__ == "__main__":
    main()
