"""Record the existing Slurm workflow without modifying jobs or experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/research_workflow"
DIRECTORY = OUTPUT / "monitoring"


def write(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def query(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(f"{command[0]} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def inspect(jobs):
    queue = query(["squeue", "--noheader", "--user", os.environ["USER"],
                   "--format=%i|%T|%r|%M|%l|%j"])
    rows = [line.strip().split("|", 5) for line in queue.splitlines() if line.strip()]
    active = {row[0]: dict(zip(("state", "reason", "elapsed", "limit", "name"), row[1:]))
              for row in rows if row[0] in jobs.values()}
    accounting = query(["sacct", "-X", "--jobs", ",".join(jobs.values()),
                        "--starttime=2026-10-16", "--noheader", "--parsable2",
                        "--format=JobIDRaw,State,ExitCode,Elapsed,Start,End"])
    failed = [line for line in accounting.splitlines() if line.strip() and
              line.split("|")[1].split()[0].rstrip("+") in {
                  "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "BOOT_FAIL",
                  "DEADLINE", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED"}]
    receipts = {}
    logs = {}
    for stage, job in jobs.items():
        receipt = OUTPUT / f"{stage}.json"
        if receipt.exists():
            receipts[stage] = json.loads(receipt.read_text())
        log = OUTPUT / f"slurm-{job}-{stage}.log"
        if log.exists():
            stat = log.stat()
            with log.open("rb") as stream:
                stream.seek(max(0, stat.st_size - 6000))
                tail = stream.read().decode(errors="replace")
            logs[stage] = {"size": stat.st_size, "mtime": stat.st_mtime, "tail": tail}
    priority_only = bool(active) and not failed and all(
        row["state"] == "PENDING" and row["reason"] in {"Priority", "Dependency"}
        for row in active.values()
    ) and any(row["reason"] == "Priority" for row in active.values())
    return {"queue": active, "accounting": accounting, "receipts": receipts,
            "logs": logs, "failed_accounting": failed, "priority_only": priority_only}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--next-check", required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    next_check = datetime.fromisoformat(args.next_check)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (DIRECTORY / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(DIRECTORY / "process.json", {"pid": os.getpid(), "deadline": deadline.isoformat(),
              "started_at": datetime.now(timezone.utc).isoformat(), "read_only": True})
        while True:
            now = datetime.now(timezone.utc)
            if now < min(next_check, deadline):
                time.sleep(min(60, (min(next_check, deadline) - now).total_seconds()))
                continue
            jobs = json.loads((OUTPUT / "native_chain.json").read_text())["jobs"]
            observation = {"observed_at": now.isoformat()}
            try:
                observation.update(inspect(jobs))
            except Exception as error:
                observation["query_error"] = f"{type(error).__name__}: {error}"
            finished = datetime.now(timezone.utc) >= deadline
            interval = 2.5 if observation.get("priority_only") else 2
            next_check = min(datetime.now(timezone.utc) + timedelta(hours=interval), deadline)
            observation.update(deadline=deadline.isoformat(), next_check=next_check.isoformat(),
                               monitoring_finished=finished, interval_hours=interval)
            stamp = now.strftime("%Y%m%dT%H%M%SZ")
            write(DIRECTORY / f"{stamp}.json", observation)
            write(DIRECTORY / "latest.json", observation)
            print(json.dumps({key: observation[key] for key in (
                "observed_at", "next_check", "monitoring_finished")}), flush=True)
            if finished:
                return


if __name__ == "__main__":
    main()
