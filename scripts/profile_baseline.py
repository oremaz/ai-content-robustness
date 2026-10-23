"""Measure the real Qwen baseline while retaining its checkpointed progress."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from scripts.research_chain import OUTPUT, write_record


def generation_progress(root):
    from ai_content_detector.rl_evasion.checkpointing import load_checkpoint
    result = []
    for path in sorted((Path(root) / "base_generation/generation").glob("*.pt")):
        state = load_checkpoint(path)
        result.append({"path": str(path), "completed": len(state["texts"]),
                       "requested": len(state["config"]["prompts"])})
    return result


def run(continuation_job, seconds=900):
    from ai_content_detector.rl_evasion.checkpointing import StopRequest
    from scripts.research_workflow import run_stage

    started = time.monotonic()
    before = generation_progress(OUTPUT / "defender_comparison")
    timer = threading.Timer(seconds, lambda: os.kill(os.getpid(), signal.SIGUSR1))
    timer.daemon = True
    status = "failed"
    try:
        with StopRequest():
            timer.start()
            try:
                run_stage("comparison_baseline")
                status = "completed"
            except SystemExit as error:
                if error.code != 75:
                    raise
                receipt = json.loads((OUTPUT / "comparison_baseline.json").read_text())
                if receipt.get("status") != "paused" or receipt.get("slurm_job_id") != os.environ["SLURM_JOB_ID"]:
                    raise RuntimeError("Pilot stopped without its saved pause receipt") from error
                status = "paused"
    finally:
        timer.cancel()
        elapsed = time.monotonic() - started
        after = generation_progress(OUTPUT / "defender_comparison")
        produced = sum(row["completed"] for row in after) - sum(row["completed"] for row in before)
        total_left = 1500 - sum(row["completed"] for row in after)
        write_record(OUTPUT / "baseline_profile.json", {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ["SLURM_JOB_ID"], "continuation_job": str(continuation_job),
            "status": status, "elapsed_seconds": elapsed, "new_completions": produced,
            "generation": after, "seconds_per_completion_including_setup": elapsed / produced if produced else None,
            "remaining_generation_seconds_at_observed_rate": elapsed * total_left / produced if produced else None,
            "note": "Generation rate includes setup; seven-detector scoring is not timed by this estimate.",
        })
    limit = "00:10:00" if status == "completed" else "04:00:00"
    qos = "qos_gpu_h100-dev" if status == "completed" else "qos_gpu_h100-t3"
    subprocess.run(["scontrol", "update", f"JobId={continuation_job}", f"QOS={qos}",
                    f"TimeLimit={limit}", "NumCPUs=32", "CPUsPerTask=32", "MinCPUsNode=32",
                    "MinMemoryCPU=5000",
                    f"Dependency=afterany:{os.environ['SLURM_JOB_ID']}"], check=True)
    subprocess.run(["scontrol", "release", str(continuation_job)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-job", type=int, required=True)
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    run(args.continuation_job, args.seconds)
