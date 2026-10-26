"""Submit diagnostic-gated, matched-budget SD1.5 attacker ablations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import traceback

from ai_content_detector.rl_evasion.image_evasion import ablation, research
from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
from ai_content_detector.rl_evasion.config import ImageEvasionConfig, seed_everything
from scripts import image_experiments

ROOT = image_experiments.ROOT


def stages(manifest):
    pilots = ["pilot-" + name for name in manifest["variants"]]
    return {"diagnostic": [], **{stage: ["diagnostic"] for stage in pilots},
            "long": pilots, "test": ["long"]}


def create(parent_path):
    parent_path = Path(parent_path).resolve()
    parent = json.loads(parent_path.read_text())
    previous = Path(parent["directory"])
    warmup_root = image_experiments.receipt_directory(parent, "warmup")
    if json.loads((warmup_root / "warmup.json").read_text())["status"] != "completed":
        raise ValueError("A completed warmed defender is required")
    data_path = previous / "partitions.json"
    ablation.split_rows(json.loads(data_path.read_text()))
    warmup = previous / "warmup/completed.pt"
    recorded = json.loads((previous / "measurements/warmup_on_base.json").read_text())["checkpoint_sha256"]
    if research.file_hash(warmup) != recorded:
        raise ValueError("Warmed defender identity changed")
    path = image_experiments.create_manifest()
    manifest = json.loads(path.read_text())
    root = Path(manifest["directory"])
    (root / "warmup").symlink_to(previous / "warmup", target_is_directory=True)
    config = asdict(ImageEvasionConfig(**parent["attacker_config"]))
    config.update(dda_checkpoint=str(warmup), kl_coeff=0.0)
    calibration_path = previous / "reference_reward_calibration.json"
    manifest.update(parent_manifest=str(parent_path), parent_directory=str(previous), partitions=str(data_path),
                    partitions_sha256=research.file_hash(data_path), warmup_sha256=recorded,
                    attacker_config=config, variants=ablation.pilot_variants(), jobs={},
                    reference_scale=json.loads(calibration_path.read_text())["scale"],
                    reference_calibration_sha256=research.file_hash(calibration_path),
                    minimum_clip_fraction=0.95,
                    protocol={"pilot_images": 800, "long_image_budgets": [6400, 12800],
                              "selection": "validation only; CLIP >= 95% of initial; evasion rate, mean evasion, CLIP",
                              "test_access": "only initial and validation-selected long-run checkpoint after training",
                              "seed": 42, "regular_h100_qos": True, "community_forensics_used": False,
                              "reference_scale": "same frozen training-only probe scale across all variants",
                              "sampling": "same float32 trajectory DDIM sampler for training, validation and test"})
    research.write_json(path, manifest)
    report(path)
    return path


def submission_command(path, manifest, stage):
    parents = stages(manifest)[stage]
    duration = "12:00:00" if stage == "long" else "02:00:00" if stage.startswith("pilot-") else "01:00:00"
    command = ["sbatch", "--parsable", "--hold", "--qos=qos_gpu_h100-t3", f"--time={duration}",
               "--signal=USR1@600", "--requeue", "--kill-on-invalid-dep=yes", "--cpus-per-task=16",
               f"--job-name=dda-ablate-{stage}", f"--output={manifest['directory']}/slurm-%j-{stage}.log"]
    if parents:
        command.append("--dependency=afterok:" + ":".join(manifest["jobs"][parent]["job_id"] for parent in parents))
    return command + [str(ROOT / "scripts/run_image_slurm.sh"), str(path), stage, "image_attacker_ablation.py"]


def submit(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest["jobs"]:
        raise ValueError("Ablations already submitted")
    submitted = []
    try:
        for stage in stages(manifest):
            command = submission_command(path, manifest, stage)
            job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";", 1)[0]
            if not job.isdigit():
                raise RuntimeError(f"Invalid Slurm response: {job}")
            submitted.append(job)
            manifest["jobs"][stage] = {"job_id": job, "command": command}
            research.write_json(path, manifest)
            print(stage, job, flush=True)
        validation = {}
        for stage, job in manifest["jobs"].items():
            status = subprocess.check_output(["scontrol", "show", "job", "-o", job["job_id"]], text=True)
            if "JobState=PENDING" not in status or "Reason=JobHeldUser" not in status:
                raise RuntimeError(f"New job was not held: {status}")
            for parent in stages(manifest)[stage]:
                if f"afterok:{manifest['jobs'][parent]['job_id']}" not in status:
                    raise RuntimeError(f"Dependency absent: {stage}, {parent}")
            validation[stage] = status.strip()
        manifest["dependency_validation"] = validation
        research.write_json(path, manifest)
        for job in submitted:
            subprocess.run(["scontrol", "release", job], check=True)
    except BaseException:
        if submitted:
            subprocess.run(["scancel", *submitted], check=True)
        raise
    report(path)


def report(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    root = Path(manifest["directory"])
    if manifest.get("retired_pilot_artifacts"):
        from scripts.report_image_attacker_summary import report as concise_report
        concise_report(path)
        return
    with (root / "ablation-report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lines = ["# SD1.5 attacker improvement experiments", "", f"Run: `{root.name}`.", "",
                 "All pilots use 800 training images and the same warmed DDA. Validation selects checkpoints; test prompts are excluded.",
                 "The long run starts from initial SD1.5, using the validation-selected pilot configuration, and reaches 12,800 images with a measurement at 6,400.",
                 "The CLIP floor is 95% of the initial model's mean validation CLIP score. Test is accessed only after long-run selection.", "",
                 "| Stage | Job | Status |", "| --- | --- | --- |"]
        for stage in stages(manifest):
            receipt = root / f"{stage}.json"
            record = json.loads(receipt.read_text()) if receipt.exists() else {}
            lines.append(f"| {stage} | {manifest['jobs'].get(stage, {}).get('job_id', 'not submitted')} | {record.get('status', 'pending')} |")
            if record.get("error"):
                lines += ["", f"{stage}: `{record['error']}`", ""]
        diagnostic = root / "diagnostic_metrics.json"
        if diagnostic.exists():
            values = json.loads(diagnostic.read_text())
            lines += ["", "## Optimizer diagnostic", "", f"Maximum replay log-probability error: `{values['max_replay_logprob_error']:.8g}`.",
                      f"Initial policy-gradient norm: `{values['initial_gradient_norm']:.8g}`."]
        lines += ["", "## Validation-selected pilots", "",
                  "| Configuration | Images at selected checkpoint | Evasion | Mean evasion reward | CLIP |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for name in list(manifest["variants"]) + ["long"]:
            selected = root / name / "selected.json"
            if selected.exists():
                v = json.loads(selected.read_text())
                lines.append(f"| {name} | {v['images_seen']} | {v['evasion_rate']:.1%} | {v['mean_evasion']:.5f} | {v['clip_mean']:.5f} |")
        curves = []
        for name in list(manifest["variants"]) + ["long"]:
            for metrics in sorted((root / name / "validation").glob("epoch-*/metrics.json")):
                value = json.loads(metrics.read_text())
                curves.append({"variant": name, "epoch": int(metrics.parent.name.split("-")[-1]),
                               **{key: value[key] for key in ("evasion_rate", "mean_evasion", "clip_mean")}})
        if curves:
            with (root / "validation_curves.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
                writer.writeheader()
                writer.writerows(curves)
            lines += ["", "All validation checkpoints are retained in `validation_curves.csv`; optimizer and reward traces are in each run's `training_history.json`."]
        final = root / "test_results.json"
        if final.exists():
            lines += ["", "## Held-out test", "", "```json", json.dumps(json.loads(final.read_text()), indent=2), "```"]
        lines += ["", "Eight pilots use matched image budgets; their numbers of rollout rounds and updates differ.",
                  "Grouped sampling replaces historical prompt normalization. Logit scaling uses only saved training images.",
                  "Reference weights 0, 0.03 and 0.10 retain evasion/CLIP weights 0.64/0.26; joint advantage normalization follows weighting.",
                  "One seed is used. Pilot selection is exploratory; the old published evaluation is not used for selection.", ""]
        (root / "report.md").write_text("\n".join(lines))
        repository = Path(manifest["repository"])
        (repository / "docs/image_attacker_ablation_results.md").write_text(
            "# Attacker improvement experiments\n\n[Live experiment report](../" + str((root / "report.md").relative_to(repository)) + ").\n")
    from scripts.report_image_attacker_summary import report as concise_report
    concise_report(path)


def run(path, stage):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    root = Path(manifest["directory"])
    job = os.environ.get("SLURM_JOB_ID")
    if manifest["jobs"][stage]["job_id"] != job:
        raise ValueError("Slurm job identity changed")
    for name, expected in manifest["code_sha256"].items():
        if research.file_hash(Path(manifest["source"]) / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    if research.file_hash(manifest["partitions"]) != manifest["partitions_sha256"]:
        raise ValueError("Frozen splits changed")
    if research.file_hash(root / "warmup/completed.pt") != manifest["warmup_sha256"]:
        raise ValueError("Frozen warmed defender changed")
    reference = Path(manifest["parent_directory"]) / "reference_reward_calibration.json"
    if research.file_hash(reference) != manifest["reference_calibration_sha256"]:
        raise ValueError("Frozen reference calibration changed")
    checkpoint = manifest["resources"]["checkpoint"]["files"]["DDA_ckpt.pth"]
    if research.file_hash(checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("Official DDA base checkpoint changed")
    for parent in stages(manifest)[stage]:
        receipt = json.loads((root / f"{parent}.json").read_text())
        if receipt.get("status") != "completed" or receipt.get("job_id") != manifest["jobs"][parent]["job_id"]:
            raise ValueError(f"Dependency did not complete: {parent}")
    with (root / f"{stage}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = {"status": "running", "job_id": job, "started_at": datetime.now(timezone.utc).isoformat(),
                   "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0"))}
        target = root / f"{stage}.json"
        research.write_json(target, receipt)
        try:
            seed_everything(42)
            with StopRequest() as stop:
                if stage == "diagnostic":
                    ablation.diagnostic(manifest, stop)
                elif stage.startswith("pilot-"):
                    ablation.pilot(manifest, stage.removeprefix("pilot-"), stop)
                elif stage == "long":
                    ablation.long_run(manifest, stop)
                else:
                    ablation.final_test(manifest, stop)
        except CheckpointPause as error:
            receipt.update(status="paused", error=str(error))
            research.write_json(target, receipt)
            report(path)
            if receipt["restart_count"] >= manifest["max_restarts"]:
                raise RuntimeError("Ablation restart limit exceeded") from error
            subprocess.run(["scontrol", "requeue", job], check=True)
        except BaseException as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            research.write_json(target, receipt)
            report(path)
            traceback.print_exc()
            raise
        else:
            receipt.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
            research.write_json(target, receipt)
            try:
                report(path)
            except BaseException as error:
                receipt.update(status="failed", error=f"Reporting failed: {type(error).__name__}: {error}")
                research.write_json(target, receipt)
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    create_parser = commands.add_parser("prepare")
    create_parser.add_argument("parent_manifest", type=Path)
    for mode in ("submit", "report", "run"):
        command = commands.add_parser(mode)
        command.add_argument("manifest", type=Path)
        if mode == "run":
            command.add_argument("stage")
    args = parser.parse_args()
    if args.mode == "prepare":
        print(create(args.parent_manifest))
    elif args.mode == "submit":
        submit(args.manifest)
    elif args.mode == "report":
        report(args.manifest)
    else:
        run(args.manifest, args.stage)


if __name__ == "__main__":
    main()
