"""Submit and resume the agreed RoBERTa, Gemma and DivEye races."""

from datetime import datetime, timezone
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil

from scripts.research_chain import ROOT, OUTPUT, write_record

DEFENDERS = {"roberta": "roberta", "gemma": "decoder_classifier", "diveye": "diveye"}
METRIC_COLUMNS = ["race", "round", "phase", "attacker_updates", "defender_round", "recorded_at", "detector",
                  "accuracy", "f1_macro", "roc_auc", "ai_detection_rate", "human_false_positive_rate",
                  "attack_success_rate", "threshold", "num_ai", "num_human", "num_generated",
                  "num_ineligible_pairs", "conditional_attack_success_rate",
                  "valid_evasion_fraction_of_all_outputs", "tpr_at_validation_1pct_fpr",
                  "test_fpr_at_validation_threshold", "validation_threshold",
                  "mean_semantic_similarity", "mean_ai_words", "mean_human_words"]


def write_metrics(path, rows):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_metrics(path, name, round_number, phase, measurement, steps):
    path = Path(path)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        keys = ("race", "round", "phase", "detector")
        existing = {tuple(row[key] for key in keys): row for row in rows}
        metadata = {"race": name, "round": round_number, "phase": phase,
                    "attacker_updates": measurement.get("attacker_updates", (round_number - (phase == "before")) * steps),
                    "defender_round": round_number - (phase != "after_defense"),
                    "recorded_at": datetime.now(timezone.utc).isoformat()}
        diagnostics = {key: measurement.get(key) for key in
                       ("mean_semantic_similarity", "mean_ai_words", "mean_human_words")}
        for record in measurement["records"]:
            values = {**metadata, **record, **diagnostics}
            row = {key: "" if values.get(key) is None else str(values[key]) for key in METRIC_COLUMNS}
            identity = tuple(row[key] for key in keys)
            if identity in existing:
                if any(row[key] != existing[identity][key] for key in METRIC_COLUMNS if key != "recorded_at"):
                    raise ValueError("Conflicting metrics for an already recorded race phase")
                continue
            rows.append(row)
            existing[identity] = row
        write_metrics(path, rows)


def inputs():
    from ai_content_detector.rl_evasion.defender_comparison import workflow
    from ai_content_detector.rl_evasion.defender_comparison.protocol import digest
    from scripts.research_workflow import validate_prerequisites

    receipt = json.loads((OUTPUT / "comparison_warmup.json").read_text())
    if receipt.get("status") != "completed":
        raise RuntimeError("The seven-defender warm-up must be completed")
    plan = workflow.load_plan()
    checkpoints = workflow.warm_checkpoints()
    identity = workflow.checkpoint_identity({name: checkpoints[name] for name in DEFENDERS.values()})
    files = ("base_pairs.json", "prompt_manifest.json", "validation_thresholds.json", "readiness_target.json")
    hashes = {name: hashlib.sha256((workflow.OUTPUT / name).read_bytes()).hexdigest() for name in files}
    return {"baseline_plan_sha256": digest(plan), "initial_defenders": identity, "input_sha256": hashes,
            "prerequisites": validate_prerequisites("comparison_race_decoder")}, plan


def submit(pilot=False, attacker_only=False):
    identity, plan = inputs()
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = OUTPUT / "defender_comparison/races_100" / identifier / "manifest.json"
    protocol = {"rounds": 10, "attacker_steps_per_round": 300, "initialization": "base_Qwen",
                "defenders": DEFENDERS, "evaluation_prompts": 500, "defender_training_prompts": 1000,
                "seed": 42, "execution": {"generation_batch_size": 8,
                "batch_semantic_evaluation": True, "reuse_before_evaluation": True,
                "diagnostic_samples": 100,
                "attacker_training": {"learning_rate": 5e-5, "lr_scheduler_type": "constant_with_warmup",
                                      "warmup_steps": 50}}}
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                "inputs": identity, "baseline_plan": plan, "qos": "qos_gpu_h100-t3",
                "time_limit": "20:00:00", "gpu_limit": 3, "max_restarts": 100,
                "metrics_file": str(path.parent / "metrics.csv"), "held": True, "jobs": {}}
    if pilot:
        manifest.update(qos="qos_gpu_h100-dev", time_limit="02:00:00", pause_after_round=3)
    if attacker_only:
        manifest.pop("pause_after_round", None)
        manifest["pause_after_attacker_round"] = 1
    write_record(path, manifest)
    write_metrics(Path(manifest["metrics_file"]), [])
    submit_jobs(path, manifest)
    return path


def submit_jobs(path, manifest):
    protocol = manifest["protocol"]
    try:
        for name in protocol["defenders"]:
            directory = path.parent / name
            directory.mkdir(parents=True, exist_ok=True)
            job_prefix = manifest.get("job_prefix") or (
                "defend" if not protocol["attacker_steps_per_round"] else f"race{protocol['attacker_steps_per_round']}")
            command = ["sbatch", "--parsable", "--hold", f"--qos={manifest['qos']}", f"--time={manifest['time_limit']}",
                       "--signal=USR1@600", "--requeue", f"--cpus-per-task={manifest.get('cpus_by_branch', {}).get(name, 32)}",
                       f"--job-name={job_prefix}-{name}",
                       f"--output={directory}/slurm-%j.log", str(ROOT / "scripts/run_research_slurm.sh"),
                       manifest.get("runner_mode", "--architecture-race"), str(path), name]
            if manifest.get("continuation", {}).get("prepare_on_start"):
                source_job = manifest["continuation"]["source_jobs"][name]
                command[2:2] = [f"--dependency=afterok:{source_job}", "--kill-on-invalid-dep=yes"]
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


def submit_batched_pilot(beta, effective_batch, generations):
    if not math.isfinite(beta) or beta <= 0 or effective_batch < 1 or generations < 2 or effective_batch % generations:
        raise ValueError("Require a positive KL coefficient and complete prompt groups in the effective batch")
    active = subprocess.check_output(["squeue", "--noheader", "--user", os.environ["USER"], "--format=%i"],
                                     text=True).split()
    if active:
        raise ValueError("Inspect active jobs before adding three independent H100 allocations")
    identity, plan = inputs()
    candidates = [size for size in (2, 4, 8, 16, 32) if size <= effective_batch and effective_batch % size == 0]
    if not candidates:
        raise ValueError("No supported physical microbatch divides the requested effective batch")
    initial_microbatch = min(8, candidates[-1])
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = OUTPUT / "defender_comparison/races_100" / identifier / "manifest.json"
    protocol = {"rounds": 1, "attacker_steps_per_round": 500, "initialization": "base_Qwen",
                "defenders": DEFENDERS, "evaluation_prompts": 500, "defender_training_prompts": 1000,
                "seed": 42, "execution": {"generation_batch_size": 8,
                "batch_semantic_evaluation": True, "reuse_before_evaluation": True,
                "diagnostic_samples": 100, "microbatch_candidates": candidates,
                "attacker_training": {"learning_rate": 3e-5, "lr_scheduler_type": "warmup_stable_decay",
                                      "warmup_steps": 50,
                                      "lr_scheduler_kwargs": {"num_decay_steps": 50, "min_lr_ratio": 1 / 3},
                                      "grpo_beta": beta, "grpo_num_generations": generations,
                                      "per_device_train_batch_size": initial_microbatch,
                                      "gradient_accumulation_steps": effective_batch // initial_microbatch}}}
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                "inputs": identity, "baseline_plan": plan, "qos": "qos_gpu_h100-dev",
                "time_limit": "02:00:00", "gpu_limit": 3, "max_restarts": 100,
                "metrics_file": str(path.parent / "metrics.csv"), "held": True, "jobs": {},
                "pause_after_attacker_round": 1}
    write_record(path, manifest)
    write_metrics(Path(manifest["metrics_file"]), [])
    submit_jobs(path, manifest)
    return path


def source_attacker(source, name):
    from ai_content_detector.rl_evasion.checkpointing import load_checkpoint

    previous_steps = source["protocol"]["attacker_steps_per_round"]
    training = source["protocol"]["execution"]["attacker_training"]
    directory = Path(source["jobs"][name]["directory"])
    receipt = json.loads((directory / "receipt.json").read_text())
    attacker = directory / "round_1/attacker"
    checkpoint = attacker / f"checkpoint-{previous_steps}"
    for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "restart_complete.pt",
                     "adapter_model.safetensors", "trainer_state.json"):
        if not (checkpoint / filename).is_file():
            raise ValueError(f"Missing native continuation state: {checkpoint / filename}")
    native = json.loads((checkpoint / "trainer_state.json").read_text())
    context = load_checkpoint(attacker / "restart_context.pt")
    state = load_checkpoint(directory / "race_restart.pt")
    scheduler = load_checkpoint(checkpoint / "scheduler.pt")
    if (receipt.get("status") != "attacker_complete" or state["round"] != 0
            or state["phase"] != "generate_defender" or state["history"]
            or state["attacker_updates"] != previous_steps
            or native["global_step"] != previous_steps or native["max_steps"] != previous_steps
            or not context.get("completed") or context["max_steps"] != previous_steps):
        raise ValueError(f"Source attacker is not complete at the recorded budget: {name}")
    if any(context["config"].get(key, {} if key == "lr_scheduler_kwargs" else None) != value
           for key, value in training.items()):
        raise ValueError(f"Source attacker schedule differs from its manifest: {name}")
    rates = scheduler["_last_lr"]
    if (scheduler["last_epoch"] != previous_steps or not rates
            or any(rate != training["learning_rate"] for rate in rates)):
        raise ValueError(f"Source scheduler is not at its completed constant-rate position: {name}")
    return directory, checkpoint, context, state


def copy_attacker(prepared, destination, steps, *, execution=None, rounds=None, output_directory=None):
    from ai_content_detector.rl_evasion.checkpointing import atomic_save

    directory, checkpoint, context, state = prepared
    attacker = destination / "round_1/attacker"
    shutil.copytree(checkpoint, attacker / checkpoint.name)
    shutil.copytree(directory / "round_1/before", destination / "round_1/before")
    shutil.copytree(directory / "round_1/attacker/diagnostics", attacker / "diagnostics")
    context["config"]["output_dir"] = str((output_directory or destination) / "round_1/attacker")
    context.update(max_steps=steps, completed=False)
    if execution is not None:
        context["config"].update(execution["attacker_training"])
        state["config"]["plan"]["execution"] = execution
    if rounds is not None:
        state["config"]["rounds"] = rounds
        state["config"]["plan"]["race_rounds"] = rounds
    atomic_save(context, attacker / "restart_context.pt")
    state["config"]["attacker_steps"] = steps
    state["config"]["plan"]["attacker_steps_per_round"] = steps
    # This phase records its full horizon once, including the resumed updates.
    state.update(phase="attack", attacker_updates=0, metrics={"before": state["metrics"]["before"]})
    atomic_save(state, destination / "race_restart.pt")
    shutil.copy2(directory / "reuse_decision.json", destination / "reuse_decision.json")
    hashes = {}
    for original in checkpoint.iterdir():
        if original.is_file():
            with original.open("rb") as stream:
                hashes[original.name] = hashlib.file_digest(stream, "sha256").hexdigest()
            with (attacker / checkpoint.name / original.name).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != hashes[original.name]:
                    raise ValueError(f"Native checkpoint copy differs: {original}")
    return {"path": str(checkpoint), "sha256": hashes}


def extend_attacker(path, steps):

    path = Path(path).resolve()
    source = json.loads(path.read_text())
    previous_steps = source["protocol"]["attacker_steps_per_round"]
    training = source["protocol"].get("execution", {}).get("attacker_training", {})
    if (steps <= previous_steps or source.get("pause_after_attacker_round") != 1
            or source["protocol"]["defenders"] != DEFENDERS
            or set(source["jobs"]) != set(DEFENDERS)):
        raise ValueError("Extension requires a larger budget for the three first-round attackers")
    if (training.get("lr_scheduler_type") != "constant_with_warmup"
            or not 0 <= training.get("warmup_steps", previous_steps) < previous_steps):
        raise ValueError("Only a completed constant-rate schedule can extend without changing its trajectory")
    active = subprocess.check_output(["squeue", "--noheader", "--user", os.environ["USER"], "--format=%i"],
                                     text=True).split()
    if active:
        raise ValueError("Inspect active jobs before adding three independent GPU allocations")
    identity, plan = inputs()
    if identity != source["inputs"] or plan != source["baseline_plan"]:
        raise ValueError("Race inputs or warmed defenders changed since the source run")
    prepared = {name: source_attacker(source, name) for name in source["jobs"]}

    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = OUTPUT / "defender_comparison/races_100" / identifier / "manifest.json"
    manifest = copy.deepcopy(source)
    manifest["protocol"]["attacker_steps_per_round"] = steps
    manifest.update(created_at=datetime.now(timezone.utc).isoformat(), jobs={}, held=True,
                    qos="qos_gpu_h100-dev", time_limit="02:00:00",
                    metrics_file=str(target.parent / "metrics.csv"),
                    continuation={"source_manifest": str(path), "source_steps": previous_steps,
                                  "target_steps": steps, "checkpoints": {}})
    for key in ("released_at", "resource_verification", "validation", "previous_submissions",
                "diagnostic_runtime_estimate"):
        manifest.pop(key, None)
    for name, values in prepared.items():
        manifest["continuation"]["checkpoints"][name] = copy_attacker(values, target.parent / name, steps)
    write_record(target, manifest)
    write_metrics(Path(manifest["metrics_file"]), [])
    submit_jobs(target, manifest)
    return target


def submit_followup(path):
    path = Path(path).resolve()
    source = json.loads(path.read_text())
    protocol = copy.deepcopy(source["protocol"])
    expected = {"learning_rate": 5e-5, "lr_scheduler_type": "constant_with_warmup", "warmup_steps": 50}
    if (source.get("pause_after_attacker_round") != 1 or protocol["attacker_steps_per_round"] != 600
            or protocol["defenders"] != DEFENDERS or set(source["jobs"]) != set(DEFENDERS)
            or protocol["execution"]["attacker_training"] != expected):
        raise ValueError("Follow-up requires the agreed three 600-step constant-rate attackers")
    active = subprocess.check_output(["squeue", "--noheader", "--user", os.environ["USER"], "--format=%i"],
                                     text=True).split()
    if set(active) - {job["job_id"] for job in source["jobs"].values()}:
        raise ValueError("Inspect other active jobs before adding the three dependent GPU allocations")
    identity, plan = inputs()
    if identity != source["inputs"] or plan != source["baseline_plan"]:
        raise ValueError("Race inputs or warmed defenders changed since the source run")
    protocol.update(rounds=3, attacker_steps_per_round=750)
    protocol["execution"]["attacker_training"] = {
        "learning_rate": 5e-5, "lr_scheduler_type": "warmup_stable_decay", "warmup_steps": 50,
        "lr_scheduler_kwargs": {"num_decay_steps": 100, "min_lr_ratio": 0.2},
    }
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = OUTPUT / "defender_comparison/races_100" / identifier / "manifest.json"
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                "inputs": identity, "baseline_plan": plan, "qos": "qos_gpu_h100-t3",
                "time_limit": "11:45:00", "gpu_limit": 3, "max_restarts": 100,
                "metrics_file": str(target.parent / "metrics.csv"), "held": True, "jobs": {},
                "continuation": {"source_manifest": str(path), "source_steps": 600, "target_steps": 750,
                                 "prepare_on_start": True,
                                 "source_jobs": {name: job["job_id"] for name, job in source["jobs"].items()}}}
    write_record(target, manifest)
    write_metrics(Path(manifest["metrics_file"]), [])
    submit_jobs(target, manifest)
    return target


def prepare_followup(manifest, name):
    continuation = manifest.get("continuation", {})
    if not continuation.get("prepare_on_start"):
        return
    directory = Path(manifest["jobs"][name]["directory"])
    marker = directory / "continuation.json"
    identity = {"source_manifest": continuation["source_manifest"], "protocol": manifest["protocol"]}
    if marker.exists():
        if json.loads(marker.read_text())["identity"] != identity:
            raise ValueError("Prepared continuation differs from the submitted protocol")
        return
    source = json.loads(Path(continuation["source_manifest"]).read_text())
    if (source["jobs"][name]["job_id"] != continuation["source_jobs"][name]
            or source["inputs"] != manifest["inputs"] or source["baseline_plan"] != manifest["baseline_plan"]):
        raise ValueError("Continuation source or inputs changed since submission")
    prepared = source_attacker(source, name)
    staging = directory / ".continuation-staging"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        proof = copy_attacker(prepared, staging, continuation["target_steps"],
                              execution=manifest["protocol"]["execution"],
                              rounds=manifest["protocol"]["rounds"], output_directory=directory)
        # Training starts only after this preparation's marker is committed.
        for entry in ("round_1", "race_restart.pt", "reuse_decision.json"):
            destination = directory / entry
            if destination.is_dir():
                shutil.rmtree(destination)
            os.replace(staging / entry, destination)
        write_record(marker, {"identity": identity, "checkpoint": proof})
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def release(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if not manifest["held"]:
        raise ValueError("Race jobs are already released")
    subprocess.run(["scontrol", "release", ",".join(job["job_id"] for job in manifest["jobs"].values())], check=True)
    manifest.update(held=False, released_at=datetime.now(timezone.utc).isoformat())
    write_record(path, manifest)
    write_record(OUTPUT / "architecture_races.json", {"manifest": str(path), "jobs": manifest["jobs"],
                                                     "protocol": manifest["protocol"]})


def resume(path, through_round=None):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if through_round is not None:
        if not manifest.get("pause_after_round", 0) <= through_round <= manifest["protocol"]["rounds"]:
            raise ValueError("Continuation target must extend the pilot within the planned ten rounds")
        manifest["pause_after_round"] = through_round
        manifest.pop("pause_after_attacker_round", None)
    active = subprocess.check_output(["squeue", "--noheader", "--user", os.environ["USER"], "--format=%i"], text=True).split()
    if any(job["job_id"] in active for job in manifest["jobs"].values()):
        raise ValueError("A race job is still active; do not launch duplicate writers")
    target = manifest.get("pause_after_round", manifest["protocol"]["rounds"])
    previous = {name: dict(job) for name, job in manifest["jobs"].items()}
    submitted = []
    try:
        for name, job in manifest["jobs"].items():
            receipt_path = Path(job["directory"]) / "receipt.json"
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
            if manifest.get("pause_after_attacker_round") and receipt.get("status") == "attacker_complete":
                continue
            if receipt.get("completed_rounds", 0) >= target:
                continue
            command = list(job["command"])
            identifier = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";", 1)[0]
            if not identifier.isdigit():
                raise ValueError("Unexpected continuation Slurm response")
            submitted.append(identifier)
            job.update(job_id=identifier)
        manifest.setdefault("previous_submissions", []).append(previous)
        manifest.update(held=True)
        write_record(path, manifest)
        if submitted:
            subprocess.run(["scontrol", "release", ",".join(submitted)], check=True)
        manifest.update(held=False)
        write_record(path, manifest)
    except BaseException:
        if submitted:
            subprocess.run(["scancel", *submitted], check=True)
        raise


def replace_and_release(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if not manifest["held"] or set(manifest["jobs"]) != set(DEFENDERS):
        raise ValueError("Replacement requires all three new jobs to be held")
    native_path = OUTPUT / "native_chain.json"
    native = json.loads(native_path.read_text())
    superseded = {stage: job for stage, job in native["jobs"].items()
                  if stage.startswith(("comparison_rl_", "comparison_race_"))}
    if len(superseded) != 6:
        raise ValueError("Expected exactly the six superseded standalone/race jobs")
    rows = subprocess.check_output(["squeue", "--noheader", "--jobs", ",".join(superseded.values()),
                                    "--format=%i|%u|%j|%T"], text=True)
    active = []
    for line in rows.splitlines():
        job, user, name, state = line.strip().split("|")
        if job not in superseded.values() or user != os.environ["USER"] or not name.startswith("research-comparison_"):
            raise ValueError("Superseded job identity differs from the saved research chain")
        if state != "PENDING":
            raise ValueError("A superseded job is no longer pending; inspect its checkpoint before replacement")
        active.append(job)
    if active:
        subprocess.run(["scancel", *active], check=True)
    manifest["superseded_jobs"] = superseded
    write_record(path, manifest)
    native.update(status="superseded_by_architecture_races", superseded_at=datetime.now(timezone.utc).isoformat(),
                  replacement_manifest=str(path), temporary_holds={})
    write_record(native_path, native)
    subprocess.run(["scontrol", "release", ",".join(row["job_id"] for row in manifest["jobs"].values())], check=True)
    manifest.update(held=False, released_at=datetime.now(timezone.utc).isoformat())
    write_record(path, manifest)
    write_record(OUTPUT / "architecture_races.json", {"manifest": str(path), "jobs": manifest["jobs"],
                                                     "protocol": manifest["protocol"]})


def run(path, name):
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
    from ai_content_detector.rl_evasion.defender_comparison import workflow
    from scripts.research_workflow import source_provenance
    from scripts.report_research import update_report

    path = Path(path)
    manifest = json.loads(path.read_text())
    job = manifest["jobs"][name]
    job_id = os.environ["SLURM_JOB_ID"]
    if job["job_id"] != job_id:
        raise ValueError("Race runner does not match the submitted Slurm job")
    directory = Path(job["directory"])
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = directory / "receipt.json"
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        if previous.get("status") == "completed" or (previous.get("status") == "pilot_complete" and
                previous.get("completed_rounds", 0) >= manifest.get("pause_after_round", 10)):
            return
        if previous.get("status") == "attacker_complete" and manifest.get("pause_after_attacker_round"):
            return
        receipt = {"status": "running", "slurm_job_id": job_id, "attempt": previous.get("attempt", 0) + 1,
                   "started_at": datetime.now(timezone.utc).isoformat(), "protocol": manifest["protocol"]}
        write_record(receipt_path, receipt)
        try:
            with StopRequest() as stop:
                identity, plan = inputs()
                if identity != manifest["inputs"] or plan != manifest["baseline_plan"]:
                    raise ValueError("Race inputs or warmed defenders changed since submission")
                prepare_followup(manifest, name)
                receipt.update(source_provenance())
                write_record(receipt_path, receipt)
                protocol = manifest["protocol"]
                names = protocol["defenders"]
                warm_start = manifest.get("warm_starts", {}).get(name)
                if warm_start is not None:
                    if (protocol["initialization"] != "retained_attackers" or protocol["rounds"] != 1
                            or protocol["attacker_steps_per_round"] != 0):
                        raise ValueError("Retained submissions must contain one defender update and no attacker training")
                    current = workflow.checkpoint_identity(workflow.warm_checkpoints())
                    if current != manifest["all_initial_defenders"]:
                        raise ValueError("An ensemble defender changed after submission")
                    selected = names[name]
                else:
                    if names != DEFENDERS or protocol["initialization"] != "base_Qwen":
                        raise ValueError("Race defender selection or initialization differs from the agreed protocol")
                    selected = [names[name]]
                from ai_content_detector.rl_evasion.defender_comparison.readiness import require_readiness
                require_readiness(workflow.OUTPUT, selected)
                plan = {**plan, "race_rounds": protocol["rounds"],
                        "attacker_steps_per_round": protocol["attacker_steps_per_round"],
                        "race_initialization": protocol["initialization"]}
                if "execution" in protocol:
                    plan["execution"] = protocol["execution"]

                def record(round_number, phase, measurement):
                    save_metrics(manifest["metrics_file"], name, round_number, phase, measurement,
                                 protocol["attacker_steps_per_round"])

                for journal in sorted(directory.glob("round_*/*/evaluation.json")):
                    record(int(journal.parent.parent.name.removeprefix("round_")), journal.parent.name,
                           json.loads(journal.read_text()))
                options = {"pause_after_round": manifest["pause_after_round"]} if "pause_after_round" in manifest else {}
                if "pause_after_attacker_round" in manifest:
                    options["pause_after_attacker_round"] = manifest["pause_after_attacker_round"]
                if warm_start is not None:
                    options["warm_start"] = warm_start
                result = workflow.race(plan, name, stop=stop, directory=directory,
                                       defender_names=selected, rounds=protocol["rounds"],
                                       attacker_steps=protocol["attacker_steps_per_round"], reuse_standalone=False,
                                       on_round=update_report, on_measurement=record, **options)
                write_record(directory / "results.json", result)
            receipt["completed_rounds"] = len(result["history"])
            receipt["status"] = ("attacker_complete" if result.get("attacker_only_complete") else
                                 "pilot_complete" if result.get("pilot_complete") else "completed")
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
                raise RuntimeError("Architecture race reached its restart limit")
            subprocess.run(["scontrol", "requeue", job_id], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    submission = commands.add_parser("submit")
    submission.add_argument("--attacker-only", action="store_true")
    pilot = commands.add_parser("submit-pilot")
    pilot.add_argument("--attacker-only", action="store_true")
    releasing = commands.add_parser("release")
    releasing.add_argument("manifest", type=Path)
    continuing = commands.add_parser("resume")
    continuing.add_argument("manifest", type=Path)
    continuing.add_argument("--through-round", type=int)
    extension = commands.add_parser("extend-attacker")
    extension.add_argument("manifest", type=Path)
    extension.add_argument("--steps", type=int, required=True)
    followup = commands.add_parser("submit-followup")
    followup.add_argument("manifest", type=Path)
    batched = commands.add_parser("submit-batched-pilot")
    batched.add_argument("--beta", type=float, required=True)
    batched.add_argument("--effective-batch", type=int, required=True)
    batched.add_argument("--generations", type=int, required=True)
    replacement = commands.add_parser("replace-and-release")
    replacement.add_argument("manifest", type=Path)
    execution = commands.add_parser("run")
    execution.add_argument("manifest", type=Path)
    execution.add_argument("defender", choices=(*DEFENDERS, "modernbert", "ensemble"))
    args = parser.parse_args()
    if args.mode == "submit":
        print(submit(attacker_only=args.attacker_only))
    elif args.mode == "submit-pilot":
        print(submit(pilot=True, attacker_only=args.attacker_only))
    elif args.mode == "release":
        release(args.manifest)
    elif args.mode == "resume":
        resume(args.manifest, args.through_round)
    elif args.mode == "extend-attacker":
        print(extend_attacker(args.manifest, args.steps))
    elif args.mode == "submit-followup":
        print(submit_followup(args.manifest))
    elif args.mode == "submit-batched-pilot":
        print(submit_batched_pilot(args.beta, args.effective_batch, args.generations))
    elif args.mode == "replace-and-release":
        replace_and_release(args.manifest)
    else:
        run(args.manifest, args.defender)
