"""Submit and report the official-DDA benchmark and one SD1.5 race round."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

from ai_content_detector.rl_evasion.image_evasion import research
from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
from ai_content_detector.rl_evasion.config import ImageEvasionConfig, seed_everything

ROOT = Path(__file__).resolve().parents[1]
STAGES = {"prepare": [], "benchmark": ["prepare"], "warmup": ["prepare"],
          "attack": ["warmup"], "defend": ["attack"]}


def create_manifest():
    directory = ROOT / "results/research_workflow/image_dda" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    resources = json.loads((ROOT / "results/research_workflow/image_dda/resources/provenance.json").read_text())
    prerequisites = json.loads((ROOT / "results/research_workflow/prerequisites.json").read_text())
    models = {name: prerequisites[repo_id] for name, repo_id in {
        "sd": "stable-diffusion-v1-5/stable-diffusion-v1-5", "clip": "openai/clip-vit-base-patch32",
        "prompts": "Gustavosta/Stable-Diffusion-Prompts"}.items()}
    checkpoint = resources["checkpoint"]["files"]["DDA_ckpt.pth"]
    if research.file_hash(checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("Official DDA weights changed")
    source = directory / "source"
    for folder in ("ai_content_detector", "scripts"):
        shutil.copytree(ROOT / folder, source / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    code_hashes = {str(path.relative_to(source)): research.file_hash(path) for path in source.rglob("*") if path.is_file() and path.suffix in {".py", ".ipynb"}}
    config = ImageEvasionConfig(output_dir=str(directory / "attacker"),
                               diffusion_model=models["sd"]["path"], num_prompts=1500,
                               dda_checkpoint=str(directory / "warmup/completed.pt"),
                               clip_reward_model=models["clip"]["path"],
                               dda_source=resources["Dual-Data-Alignment"]["path"])
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "directory": str(directory),
                "repository": str(ROOT), "source": str(source), "code_sha256": code_hashes,
                "notebook": str(source / "ai_content_detector/notebooks/bench-imai-artifact.ipynb"),
                "resources": resources, "models": models, "attacker_config": asdict(config),
                "protocol": {"rounds": 1, "benchmark_training": False, "warmup_origin": "official_DDA",
                             "fitting_prompts": 1000, "evaluation_prompts": 500,
                             "defender_validation_fraction": 0.2, "race_resize": 200,
                             "defender_epochs": 3, "defender_lr": 2e-5,
                             "kl_reference": "initial_SD1.5_without_adapters",
                             "kl_reduction": "mean_per_latent_coordinate_per_stochastic_transition",
                             "reference_reward": "1 / (1 + trajectory_mean_KL / fixed_probe_scale)",
                             "reference_scale_selection": "median positive per-image KL after one training-only probe epoch",
                             "inference_autocast": "bfloat16", "detector_parameter_dtype": "float32",
                             "community_forensics_used": False, "seed": 42},
                "qos": "qos_gpu_h100-t3", "max_restarts": 20, "jobs": {}}
    research.write_json(directory / "manifest.json", manifest)
    return directory / "manifest.json"


def replacement_manifest(previous_path):
    previous_path = Path(previous_path).resolve()
    previous = json.loads(previous_path.read_text())
    for stage in ("attack", "defend"):
        if (Path(previous["directory"]) / stage / "restart.pt").exists():
            raise ValueError("Refusing to reuse attacker/defense work from the former objective")
    path = create_manifest()
    manifest = json.loads(path.read_text())
    root, previous_root = Path(manifest["directory"]), Path(previous["directory"])
    manifest.update(previous_manifest=str(previous_path), submission_stages=["attack", "defend"],
                    additional_dependencies={"defend": ["benchmark"]},
                    report_document="docs/image_reference_reward_results.md")
    manifest["protocol"]["prerequisites_preserved"] = ["prepare", "benchmark", "warmup"]
    for stage in ("prepare", "benchmark", "warmup"):
        manifest["jobs"][stage] = {**previous["jobs"][stage], "adopted_from_directory": str(previous_root)}
    for name in ("partitions.json", "base_fitting", "base_evaluation", "warmup", "benchmark", "preflight.json"):
        (root / name).symlink_to(previous_root / name, target_is_directory=name in {
            "base_fitting", "base_evaluation", "warmup", "benchmark"})
    (root / "measurements").mkdir()
    for name in ("official_on_base.json", "warmup_on_base.json"):
        (root / "measurements" / name).symlink_to(previous_root / "measurements" / name)
    manifest["replaced_jobs"] = {stage: previous["jobs"][stage]["job_id"] for stage in ("attack", "defend")}
    research.write_json(path, manifest)
    return path


def receipt_directory(manifest, stage):
    return Path(manifest["jobs"].get(stage, {}).get("adopted_from_directory", manifest["directory"]))


def receipt_path(manifest, stage):
    name = manifest["jobs"].get(stage, {}).get("receipt_stage", stage)
    return receipt_directory(manifest, stage) / f"{name}.json"


def defender_manifest(parent_path):
    parent_path = Path(parent_path).resolve()
    parent = json.loads(parent_path.read_text())
    previous = Path(parent["directory"])
    config = json.loads((previous / "long/effective_config.json").read_text())
    config.pop("evasion_mode")
    if research.file_hash(previous / "warmup/completed.pt") != parent["warmup_sha256"]:
        raise ValueError("Warmed defender changed")
    if research.file_hash(parent["partitions"]) != parent["partitions_sha256"]:
        raise ValueError("Parent partitions changed")
    path = create_manifest()
    manifest = json.loads(path.read_text())
    root = Path(manifest["directory"])
    (root / "warmup").symlink_to(previous / "warmup", target_is_directory=True)
    (root / "partitions.json").symlink_to(Path(parent["partitions"]))
    manifest.update(attacker_parent_manifest=str(parent_path), attacker_config=config,
                    resources=parent["resources"], models=parent["models"],
                    submission_stages=["defend"], report_document="docs/image_defender_update_results.md",
                    jobs={"attack": {"job_id": parent["jobs"]["test"]["job_id"],
                                     "adopted_from_directory": str(previous), "receipt_stage": "test"}})
    manifest["protocol"].update(attacker_selection="same validation-selected checkpoint as parent test",
                                  evaluation="reuse parent test initial and attacked images",
                                  fitting_generated_images=1000, fitting_train_images=800,
                                  fitting_validation_images=200)
    research.write_json(path, manifest)
    return path


def checkpoint_manifest(parent_path, checkpoint_path):
    parent_path, checkpoint_path = Path(parent_path).resolve(), Path(checkpoint_path).resolve()
    parent = json.loads(parent_path.read_text())
    previous = Path(parent["directory"])
    for name in ("long", "test"):
        receipt = json.loads((previous / f"{name}.json").read_text())
        if receipt.get("status") != "completed" or receipt.get("job_id") != parent["jobs"][name]["job_id"]:
            raise ValueError("Parent training and evaluation must be complete")
    if research.file_hash(parent["partitions"]) != parent["partitions_sha256"]:
        raise ValueError("Held-out prompt partitions changed")
    if research.file_hash(previous / "warmup/completed.pt") != parent["warmup_sha256"]:
        raise ValueError("Warmed defender changed")
    config = json.loads((previous / "long/effective_config.json").read_text())
    config.pop("evasion_mode")
    checkpoint = research.load_checkpoint(checkpoint_path, ImageEvasionConfig(**config))
    if checkpoint is None or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain the recorded UNet LoRA state")
    path = create_manifest()
    manifest = json.loads(path.read_text())
    manifest.update(checkpoint_parent_manifest=str(parent_path), evaluation_checkpoint=str(checkpoint_path),
                    evaluation_checkpoint_sha256=research.file_hash(checkpoint_path), checkpoint_name=checkpoint_path.name,
                    partitions_sha256=parent["partitions_sha256"], warmup_sha256=parent["warmup_sha256"],
                    baseline_test_sha256=research.file_hash(previous / "test_results.json"),
                    attacker_config=config, resources=parent["resources"], models=parent["models"],
                    submission_stages=["evaluate"], jobs={},
                    report_document="docs/image_final_checkpoint_test_results.md")
    research.write_json(path, manifest)
    return path


def submission_command(path, manifest, stage):
    directory = Path(manifest["directory"])
    parents = STAGES.get(stage, []) + manifest.get("additional_dependencies", {}).get(stage, [])
    dependencies = []
    for parent in parents:
        job = manifest["jobs"][parent]
        proof = job.get("completion_verified")
        if proof:
            receipt = json.loads(receipt_path(manifest, parent).read_text())
            if (proof.get("state") != "COMPLETED" or proof.get("exit_code") != "0:0"
                    or proof.get("job_id") != job["job_id"] or receipt.get("status") != "completed"
                    or receipt.get("job_id") != job["job_id"]):
                raise ValueError("Completed dependency proof and receipt must match")
        else:
            dependencies.append(job["job_id"])
    command = ["sbatch", "--parsable", "--hold", "--qos=qos_gpu_h100-t3",
               f"--time={'08:00:00' if stage == 'benchmark' else '01:00:00' if stage == 'evaluate' else '04:00:00'}",
               "--signal=USR1@600", "--requeue", "--kill-on-invalid-dep=yes",
               "--cpus-per-task=16", f"--job-name=image-dda-{stage}",
               f"--output={directory}/slurm-%j-{stage}.log"]
    if dependencies:
        command.append("--dependency=afterok:" + ":".join(dependencies))
    command += [str(ROOT / "scripts/run_image_slurm.sh"), str(path), stage]
    return command


def submit(path, release=True):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    stages = manifest.get("submission_stages", list(STAGES))
    if any(stage in manifest["jobs"] for stage in stages):
        raise ValueError("This experiment has already been submitted")
    submitted = []
    try:
        for stage in stages:
            command = submission_command(path, manifest, stage)
            job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";", 1)[0]
            if not job.isdigit():
                raise RuntimeError(f"Unexpected sbatch response: {job}")
            manifest["jobs"][stage] = {"job_id": job, "command": command}
            submitted.append(job)
            research.write_json(path, manifest)
            print(stage, job, flush=True)
        if release:
            for job in submitted:
                subprocess.run(["scontrol", "release", job], check=True)
        manifest["replacement_jobs_held"] = not release
        research.write_json(path, manifest)
    except BaseException:
        if submitted:
            subprocess.run(["scancel", *submitted], check=True)
        raise
    report(path)


def report(path):
    import csv
    path = Path(path)
    manifest = json.loads(path.read_text())
    root = Path(manifest["directory"])
    if manifest.get("evaluation_checkpoint"):
        return checkpoint_report(path)
    if manifest.get("attacker_parent_manifest"):
        return defender_report(path)
    with (root / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lines = ["# DDA image experiments", "", f"Run: `{root.name}`.", "",
                 "Official DDA is evaluated without local training on ArtiFact holdout, CIFAKE and Shoes.",
                 "The SD1.5 race uses a separate warm-up followed by one attacker and one defender update.",
                 "The attacker reward is 0.64 evasion + 0.26 CLIP alignment + 0.10 reference similarity.",
                 "Reference similarity is 1 / (1 + trajectory-mean KL / scale), with a fixed scale selected by a training-only probe.",
                 "Community Forensics is not used. All production jobs use regular H100 QoS.", "",
                 "| Stage | Slurm job | State |", "| --- | --- | --- |"]
        for stage in STAGES:
            record_path = receipt_path(manifest, stage)
            receipt = json.loads(record_path.read_text()) if record_path.exists() else {}
            job = manifest["jobs"].get(stage, {}).get("job_id", "not submitted")
            lines.append(f"| {stage} | {job} | {receipt.get('status', 'submitted' if job != 'not submitted' else 'prepared')} |")
            if receipt.get("status") == "failed":
                lines.extend(["", f"Failure in {stage}: `{receipt.get('error')}`", ""])
        calibration_path = root / "reference_reward_calibration.json"
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text())
            lines += ["", "## Reference reward scale", "", f"Fixed KL scale: `{calibration['scale']:.8g}` nats per coordinate.",
                      "The median positive post-probe per-image KL maps to reference reward 0.5.",
                      "The probe uses training prompts only; its weights are discarded before the main attacker starts."]
        benchmark_path = root / "benchmark/measurements.json"
        rows = []
        if benchmark_path.exists():
            lines += ["", "## Official checkpoint benchmark", "",
                      "| Dataset | Accuracy | Macro F1 | AI AUROC | Human FPR | AI AP |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
            for name, values in json.loads(benchmark_path.read_text()).items():
                lines.append(f"| {name} | {values['accuracy']:.4f} | {values['f1_macro']:.4f} | {values['roc_auc']:.4f} | {values['human_false_positive_rate']:.4f} | {values['average_precision_ai']:.4f} |")
                rows.append({"phase": "official_benchmark", "dataset": name, **values})
        lines += ["", "## SD1.5 round", "",
                  "| Defender and saved generator outputs | AI detection | Evasion | Human FPR | AUROC |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        phases = ["official_on_base", "warmup_on_base", "warmup_on_attacked", "updated_on_attacked", "updated_on_base"]
        for phase in phases:
            target = root / "measurements" / f"{phase}.json"
            if not target.exists():
                continue
            measurement = json.loads(target.read_text())
            values = measurement["fixed_0.5"]
            lines.append(f"| {phase} | {values['ai_detection_rate']:.1%} | {values['attack_success_rate']:.1%} | {values['human_false_positive_rate']:.1%} | {values['roc_auc']:.4f} |")
            rows.append({"phase": phase, "dataset": "SD1.5 evaluation", **values})
        if rows:
            columns = sorted(set().union(*(row.keys() for row in rows)))
            with (root / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
        history_path = root / "attacker/training_history.json"
        if history_path.exists():
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            history = json.loads(history_path.read_text())
            if history:
                figure, axes = plt.subplots(1, 2, figsize=(9, 3))
                x = [row["epoch"] for row in history]
                axes[0].plot(x, [row["mean_reward"] for row in history])
                axes[0].set(xlabel="DDPO epoch", ylabel="Combined bounded reward")
                axes[1].plot(x, [row["initial_model_kl"] for row in history])
                axes[1].set(xlabel="DDPO epoch", ylabel="KL to initial SD1.5 per coordinate")
                figure.tight_layout()
                for extension in ("png", "svg"):
                    figure.savefig(root / f"attacker_training.{extension}", dpi=160)
                plt.close(figure)
                lines += ["", "![Attacker training](attacker_training.svg)"]
                components = []
                for row in history:
                    components.extend({"epoch": row["epoch"], "component": name, **values}
                                      for name, values in row.get("reward_statistics", {}).items())
                if components:
                    columns = ["epoch", "component", "mean", "min", "max", "std", "weighted_std", "weighted_range"]
                    with (root / "reward_scales.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=columns)
                        writer.writeheader()
                        writer.writerows(components)
                    lines += ["", "Per-epoch raw and weighted component ranges and standard deviations are saved in `reward_scales.csv`."]
        lines += ["", "## Protocol and interpretation", "",
                  "The benchmark uses DDA's native 336-pixel center crop, including zero padding for small images.",
                  "It reuses the notebook's seeded 15% ArtiFact sample and 20% holdout. This is an image split, not an unseen-generator split.",
                  "SD1.5 real images come only from ArtiFact. Both classes are resized to 200 pixels before DDA's crop to control resolution shortcuts.",
                  "Of the 1,000 fitting prompts, 800 train the attacker and defender; 200 are reserved for defender validation.",
                  "The 500 evaluation prompts and 500 ArtiFact holdout real images are excluded from all fitting.",
                  "Defender selection uses validation loss; evaluation uses fixed 0.5 and the warmed defender's validation-calibrated threshold.",
                  "After the defender update, the saved base and attacked images are reused exactly. No second attacker update is performed.",
                  "KL is the conditional Gaussian current-to-initial divergence, averaged over latent coordinates and stochastic transitions; it is not an image-distribution KL.",
                  "All three main reward components are bounded in [0, 1] before weighting and joint advantage normalization. The reference contribution is capped at 0.10; no additional direct KL loss is applied.",
                  "Terminal deterministic DDIM transitions are excluded from policy updates and KL. Warm-up and defense use supervised LoRA/head adaptation, not a reproduction of DDA data construction.",
                  "The official checkpoint retains the authors' COCO training; this does not establish absence of overlap with the original ArtiFact sources.",
                  "No result establishes robustness to further adaptive attacks or arbitrary unseen generators.", ""]
        research.write_json(root / "report_inputs.json", {"manifest": str(path), "metric_rows": len(rows)})
        (root / "report.md").write_text("\n".join(lines))
        document = Path(manifest["repository"]) / manifest.get("report_document", "docs/image_results.md")
        document.write_text("# DDA image experiments\n\nLatest run and live results: [experiment report](../" +
                            str((root / "report.md").relative_to(manifest["repository"])) + ").\n\n" +
                            "Production jobs, immutable partitions, saved images, metrics and curves are recorded with that run.\n")


def defender_report(path):
    manifest = json.loads(Path(path).read_text())
    root = Path(manifest["directory"])
    receipt = json.loads((root / "defend.json").read_text()) if (root / "defend.json").exists() else {}
    job = manifest["jobs"].get("defend", {}).get("job_id", "not submitted")
    lines = ["# DDA defender update after the long SD1.5 attack", "",
             f"Job `{job}`: {receipt.get('status', 'pending')}.", "",
             "The warmed defender receives one supervised LoRA/head update on 800 attacked images and 800 ArtiFact real images; 200 pairs select the checkpoint by validation loss.",
             "Training uses three epochs, learning rate 2e-5 and effective batch size 64.",
             "Before and after adaptation, evaluation reuses the same saved 500 base, 500 attacked and 500 human images from the preceding test job.", "",
             "| Defender | Generator | AI detection | Evasion | Human FPR | AUROC |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for phase, defender, generator in (("warmup_on_base", "Warmed", "Base SD1.5"),
                                       ("warmup_on_attacked", "Warmed", "Selected attacker"),
                                       ("updated_on_base", "Updated", "Base SD1.5"),
                                       ("updated_on_attacked", "Updated", "Selected attacker")):
        target = root / "measurements" / f"{phase}.json"
        if target.exists():
            values = json.loads(target.read_text())["fixed_0.5"]
            lines.append(f"| {defender} | {generator} | {values['ai_detection_rate']:.1%} | {values['attack_success_rate']:.1%} | {values['human_false_positive_rate']:.1%} | {values['roc_auc']:.4f} |")
    lines += ["", "Table threshold: 0.5. Measurements also retain the warmed defender's frozen validation threshold.",
              "The attacker is the validation-selected checkpoint already evaluated by the preceding job. No further attacker training or test-driven checkpoint selection is performed.",
              "One seed and one update; this does not establish robustness to another adaptive attack.", ""]
    if receipt.get("error"):
        lines += [f"Failure: `{receipt['error']}`", ""]
    (root / "report.md").write_text("\n".join(lines))
    repository = Path(manifest["repository"])
    (repository / manifest["report_document"]).write_text("# Image defender update\n\n[Defender report](../" +
        str((root / "report.md").relative_to(repository)) + ").\n")


def checkpoint_report(path):
    manifest = json.loads(Path(path).read_text())
    root = Path(manifest["directory"])
    receipt_path = root / "evaluate.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    job = manifest["jobs"].get("evaluate", {}).get("job_id", "not submitted")
    lines = ["# Additional held-out SD1.5 checkpoint evaluation", "",
             f"Job `{job}`: {receipt.get('status', 'pending')}.", "",
             f"Requested checkpoint: `{manifest.get('checkpoint_name', 'recorded checkpoint')}`. This job evaluates it without further training on the same 500 test prompts, noise seeds and warmed DDA as the saved baseline.", ""]
    result = root / "results.json"
    if result.exists():
        values = json.loads(result.read_text())
        lines += ["| Generator | Evasion | CLIP alignment | Human FPR | AUROC |", "| --- | ---: | ---: | ---: | ---: |"]
        for name, label in (("initial", "Initial SD1.5"), ("evaluated", "Requested checkpoint")):
            row = values[name]
            lines.append(f"| {label} | {row['evasion_rate']:.1%} | {row['clip_mean']:.4f} | {row['classification']['human_false_positive_rate']:.1%} | {row['classification']['roc_auc']:.4f} |")
        lines += ["", f"Requested-checkpoint CLIP retention on test: {values['evaluated']['clip_retention_vs_initial']:.1%}."]
    if receipt.get("error"):
        lines += ["", f"Failure: `{receipt['error']}`"]
    lines += ["", "This additional test is exploratory and does not change the previously selected checkpoint. Evasion uses threshold 0.5; the frozen validation threshold is also recorded in the results. No defender update is performed.", ""]
    (root / "report.md").write_text("\n".join(lines))
    repository = Path(manifest["repository"])
    (repository / manifest["report_document"]).write_text("# Final checkpoint test\n\n[Evaluation report](../" +
        str((root / "report.md").relative_to(repository)) + ").\n")


def run(path, stage):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    root = Path(manifest["directory"])
    job = os.environ.get("SLURM_JOB_ID")
    if manifest["jobs"][stage]["job_id"] != job:
        raise ValueError("Slurm job identity differs from the experiment manifest")
    for name, expected in manifest["code_sha256"].items():
        if research.file_hash(Path(manifest["source"]) / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    resource = manifest["resources"]["checkpoint"]["files"]["DDA_ckpt.pth"]
    if research.file_hash(resource["path"]) != resource["sha256"]:
        raise ValueError("Official DDA checkpoint changed after submission")
    for dependency in STAGES.get(stage, []) + manifest.get("additional_dependencies", {}).get(stage, []):
        receipt = json.loads(receipt_path(manifest, dependency).read_text())
        if receipt.get("status") != "completed":
            raise ValueError(f"Required stage {dependency} is not complete")
        if receipt.get("job_id") != manifest["jobs"][dependency]["job_id"]:
            raise ValueError("Dependency completion receipt belongs to another job")
    status_path = root / f"{stage}.json"
    with (root / f"{stage}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = {"stage": stage, "job_id": job, "status": "running",
                   "started_at": datetime.now(timezone.utc).isoformat(),
                   "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0"))}
        research.write_json(status_path, receipt)
        try:
            seed_everything(42)
            with StopRequest() as stop:
                if stage == "evaluate":
                    from ai_content_detector.rl_evasion.image_evasion.ablation import evaluate_checkpoint
                    evaluate_checkpoint(manifest, stop)
                else:
                    {"prepare": research.prepare, "benchmark": research.benchmark, "warmup": research.warmup,
                     "attack": research.attack, "defend": research.defend}[stage](manifest, stop)
        except CheckpointPause as error:
            receipt.update(status="paused", error=str(error))
            research.write_json(status_path, receipt)
            report(path)
            if receipt["restart_count"] >= manifest["max_restarts"]:
                raise RuntimeError("Image experiment restart limit reached") from error
            subprocess.run(["scontrol", "requeue", job], check=True)
        except BaseException as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            research.write_json(status_path, receipt)
            report(path)
            traceback.print_exc()
            raise
        else:
            receipt.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
            research.write_json(status_path, receipt)
            report(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("prepare")
    replacement = commands.add_parser("replace-attack")
    replacement.add_argument("previous_manifest", type=Path)
    defender = commands.add_parser("attach-defender")
    defender.add_argument("parent_manifest", type=Path)
    evaluation = commands.add_parser("evaluate-checkpoint")
    evaluation.add_argument("parent_manifest", type=Path)
    evaluation.add_argument("checkpoint", type=Path)
    for mode in ("submit", "report"):
        command = commands.add_parser(mode)
        command.add_argument("manifest", type=Path)
        if mode == "submit":
            command.add_argument("--held", action="store_true")
    runner = commands.add_parser("run")
    runner.add_argument("manifest", type=Path)
    runner.add_argument("stage", choices=[*STAGES, "evaluate"])
    args = parser.parse_args()
    if args.mode == "prepare":
        print(create_manifest())
    elif args.mode == "replace-attack":
        print(replacement_manifest(args.previous_manifest))
    elif args.mode == "attach-defender":
        print(defender_manifest(args.parent_manifest))
    elif args.mode == "evaluate-checkpoint":
        print(checkpoint_manifest(args.parent_manifest, args.checkpoint))
    elif args.mode == "submit":
        submit(args.manifest, release=not args.held)
    elif args.mode == "report":
        report(args.manifest)
    else:
        run(args.manifest, args.stage)


if __name__ == "__main__":
    main()
