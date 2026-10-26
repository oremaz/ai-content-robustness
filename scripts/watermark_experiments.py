"""Prepare, submit and report the four watermark families and removal settings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess

from ai_content_detector.watermarking.protocol import (
    DEFAULT_STAGES, LATENT_STAGES, STAGES, digest, file_hash, prepare_data, use_sources, validate_partitions, write_json,
)
from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
from ai_content_detector.rl_evasion.config import seed_everything

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "results/research_workflow/watermarking"


def resources():
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi, hf_hub_download
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import yaml
    import torch
    from safetensors.torch import load_file
    directory = EXPERIMENTS / "resources"
    weights = directory / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    revisions_code = {"distseal": "d0a71cd272bb29e3b49cdc965bbaf4d24f2684ac",
                      "textseal": "79c3f8001c0022ebeb8dd73f1fc86f4bc3ddc354"}
    for name, revision in revisions_code.items():
        location = directory / name
        if not location.exists():
            subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                            f"https://github.com/facebookresearch/{name}", str(location)], check=True)
            subprocess.run(["git", "-C", str(location), "checkout", "--detach", revision], check=True)
        actual = subprocess.check_output(["git", "-C", str(location), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise ValueError(f"Unexpected upstream revision: {name} {actual}")
    api = HfApi()
    revisions = {name: api.model_info(name).sha for name in (
        "yucornetto/RAR", "fun-research/TiTok", "mit-han-lab/dc-ae-f64c128-in-1.0")}
    cards = {name: yaml.safe_load((directory / f"distseal/cards/{name}.yaml").read_text())
             for name in ("rar_xl_base", "rar_xl_distseal", "maskgit_base", "uvit-h_base",
                          "uvit-h_distseal", "detector_rar", "detector_dc-ae")}
    cards["dc-ae-f64c128_base"] = {
        "model_arch": "dc-ae", "base": "dc-ae-f64c128-in-1.0",
        "checkpoint": "https://huggingface.co/mit-han-lab/dc-ae-f64c128-in-1.0/resolve/main/model.safetensors"}
    urls = sorted({v for card in cards.values() for k, v in card.items() if k in {"checkpoint", "message"}})
    def fetch(url):
        path = weights / url.rsplit("/", 1)[-1]
        if not path.exists():
            print("Downloading", path.name, flush=True)
            if "huggingface.co/" in url:
                repo_id, filename = url.split("huggingface.co/", 1)[1].split("/resolve/main/")
                revision = revisions[repo_id]
                cached = hf_hub_download(repo_id, filename, revision=revision, cache_dir=directory / "hf")
                shutil.copyfile(cached, path)
                url = url.replace("/resolve/main/", f"/resolve/{revision}/")
            else:
                session = requests.Session()
                session.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1)))
                with session.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    temporary = path.with_suffix(path.suffix + ".partial")
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(8*1024*1024):
                            stream.write(chunk)
                    temporary.replace(path)
        print("Ready", path.name, path.stat().st_size, flush=True)
        return path.name, {"url": url, "path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}
    files = dict(ThreadPoolExecutor(max_workers=3).map(fetch, urls))
    converted = weights / "dc-ae-f64c128_base.pt"
    if not converted.exists():
        torch.save({"state_dict": load_file(weights / "model.safetensors")}, converted)
    files[converted.name] = {"path": str(converted), "sha256": file_hash(converted),
                             "converted_from": files["model.safetensors"]["sha256"]}
    cards_dir = directory / "cards"
    cards_dir.mkdir(exist_ok=True)
    for name, card in cards.items():
        for key in ("checkpoint", "message"):
            if key in card:
                card[key] = str(weights / card[key].rsplit("/", 1)[-1])
        if name == "dc-ae-f64c128_base":
            card["checkpoint"] = str(converted)
        (cards_dir / f"{name}.yaml").write_text(yaml.safe_dump(card))
    provenance = {"repositories": {name: {"path": str(directory / name), "revision": subprocess.check_output(
        ["git", "-C", str(directory / name), "rev-parse", "HEAD"], text=True).strip()}
        for name in ("distseal", "textseal")}, "hf_revisions": revisions, "files": files, "cards": str(cards_dir)}
    write_json(directory / "provenance.json", provenance)
    return directory / "provenance.json"


def prepare():
    directory = EXPERIMENTS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    inputs = json.loads((ROOT / "results/research_workflow/prerequisites.json").read_text())
    qwen = inputs["Qwen/Qwen3.5-4B-Base"]
    provenance = json.loads((EXPERIMENTS / "resources/provenance.json").read_text())
    for item in provenance["files"].values():
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("Watermark checkpoint hash changed")
    model = Path(qwen["path"])
    seed = 20261026
    counts = prepare_data(ROOT, directory / "partitions.json", model, seed)
    validate_partitions(json.loads((directory / "partitions.json").read_text()))
    source = directory / "source"
    shutil.copytree(ROOT / "ai_content_detector", source / "ai_content_detector",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "notebooks", "tests"))
    (source / "scripts").mkdir()
    shutil.copy2(__file__, source / "scripts/watermark_experiments.py")
    hashes = {str(p.relative_to(source)): file_hash(p) for p in source.rglob("*.py")}
    config = {"steps": 5000, "clean_steps": 2500, "accumulation": 8, "length": 512,
              "lr": 1e-4, "clean_lr": 1e-4, "lora_rank": 64}
    manifest = {"directory": str(directory), "repository": str(ROOT), "source": str(source),
                "code_sha256": hashes, "resources": provenance, "qwen": qwen,
                "clip": inputs["openai/clip-vit-base-patch32"], "seed": seed,
                "key": secrets.randbelow(2**30), "data": str(directory / "partitions.json"),
                "data_sha256": file_hash(directory / "partitions.json"), "counts": counts,
                "text_training": config, "image_training": {"steps": 2500, "lr": 1e-4,
                "batch": 4, "lora_rank": 16}, "image_evaluation": {"n": 512, "calibration_n": 256,
                "class_zero_n": 128, "batch": 8}, "jobs": {}, "qos": "qos_gpu_h100-t3"}
    manifest["resource_code_sha256"] = {str(p): file_hash(p)
        for item in provenance["repositories"].values() for p in Path(item["path"]).rglob("*")
        if p.is_file() and p.suffix in {".py", ".yaml"}}
    manifest["resource_code_sha256"].update({str(p): file_hash(p) for p in Path(provenance["cards"]).glob("*.yaml")})
    from importlib.metadata import version
    manifest["environment"] = {name: version(name) for name in ("torch", "transformers", "peft", "diffusers", "numpy",
                                                               "onnx", "onnxsim", "omegaconf")}
    manifest["protocol_sha256"] = digest({k: manifest[k] for k in (
        "data_sha256", "qwen", "seed", "key", "text_training", "image_training", "image_evaluation")})
    write_json(directory / "manifest.json", manifest)
    os.chmod(directory / "manifest.json", 0o600)
    report(directory / "manifest.json")
    return directory / "manifest.json"


def adapt_text(parent_path):
    parent_path = Path(parent_path).resolve()
    parent = json.loads(parent_path.read_text())
    if file_hash(parent["data"]) != parent["data_sha256"]:
        raise ValueError("Parent data partitions changed")
    validate_partitions(json.loads(Path(parent["data"]).read_text()))
    directory = EXPERIMENTS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    source = directory / "source"
    shutil.copytree(ROOT / "ai_content_detector", source / "ai_content_detector",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "notebooks", "tests"))
    (source / "scripts").mkdir()
    shutil.copy2(__file__, source / "scripts/watermark_experiments.py")
    manifest = {k: parent[k] for k in ("repository", "resources", "resource_code_sha256", "qwen", "clip",
                                       "seed", "data", "data_sha256", "counts", "environment", "qos")}
    manifest.update({"directory": str(directory), "source": str(source), "parent_manifest": str(parent_path),
                     "method": "distseal_latent_text", "submission_stages": list(LATENT_STAGES), "jobs": {},
                     "distillation_stage": "latent_distill", "distilled_model": "latent_distill/model",
                     "clean_stage": "latent_clean", "code_sha256": {
                         str(p.relative_to(source)): file_hash(p) for p in source.rglob("*.py")}})
    manifest["latent_training"] = {
        "steps": 6000, "validate_every": 250, "batch": 2, "length": 256,
        "bits": 32, "width": 128, "strength": .35, "temperature": .8, "lr": .001,
        "kl_budget": .1, "minimum_bit_accuracy": .6,
        "message": [secrets.randbelow(2) for _ in range(32)],
    }
    manifest["text_training"] = {**parent["text_training"], "lr": 3e-5}
    manifest["protocol_sha256"] = digest({k: manifest[k] for k in (
        "method", "qwen", "seed", "data_sha256", "latent_training", "text_training")})
    path = directory / "manifest.json"
    write_json(path, manifest)
    os.chmod(path, 0o600)
    report(path)
    return path


def submit(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest["jobs"]:
        raise ValueError("This manifest has already been submitted")
    jobs = {}
    for stage in manifest.get("submission_stages", DEFAULT_STAGES):
        dependencies = STAGES[stage]
        hours = 1 if stage.endswith("preflight") else 20 if stage in {"distill", "latent_distill", "latent_embedder"} else 8 if stage.endswith("_clean") else 4
        # Three independent lanes keep the experiment below four simultaneous GPUs.
        quota = {"distill": "textseal", "text_clean": "text_base",
                 "rar_clean": "rar_base", "dcae_clean": "dcae_base", "latent_clean": "latent_base"}
        predecessors = list(dict.fromkeys(dependencies + ([quota[stage]] if stage in quota else [])))
        command = ["sbatch", "--parsable", "--hold", f"--qos={manifest['qos']}",
                   f"--time={hours:02}:00:00", "--signal=USR1@600", "--requeue", "--cpus-per-task=16",
                   "--kill-on-invalid-dep=yes", f"--job-name=wm-{stage}",
                   f"--output={manifest['directory']}/slurm-%j-{stage}.log"]
        if predecessors:
            command.append("--dependency=afterok:" + ":".join(jobs[p]["job_id"] for p in predecessors))
        command += [str(ROOT / "scripts/run_research_slurm.sh"), "--watermark", str(path), stage]
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        jobs[stage] = {"job_id": job, "dependencies": predecessors, "command": command}
        manifest["jobs"] = jobs
        write_json(path, manifest)
        print(stage, job, flush=True)
    report(path)
    return jobs


def report(path):
    manifest = json.loads(Path(path).read_text())
    directory = Path(manifest["directory"])
    latent = manifest.get("method") == "distseal_latent_text"
    stages = manifest.get("submission_stages", DEFAULT_STAGES)
    description = ("DistSeal-inspired text adaptation: learned hidden-state embedder and text-only message "
                   "extractor, then teacher-to-Qwen weight distillation. This is an experimental adaptation, "
                   "not an official DistSeal text implementation." if latent else
                   "TextSeal/Qwen3.5-4B-Base and DistSeal/RAR-XL and UViT-H/DC-AE. "
                   "The green-list Qwen branch was canceled and superseded by the learned latent adaptation."
                   if manifest.get("superseded_qwen_branch") else
                   "Four families: TextSeal/Qwen3.5-4B-Base, distilled Qwen, DistSeal/RAR-XL and UViT-H/DC-AE.")
    lines = ["# Watermarking experiments", "", description, "",
             "Detection thresholds are fixed from validation data; test data never select settings. "
             "Reported false-positive rates are empirical, with finite-sample uncertainty.", "",
             "| Stage | Job | Status |", "| --- | --- | --- |"]
    for stage in stages:
        receipt = directory / f"{stage}.json"
        info = json.loads(receipt.read_text()) if receipt.exists() else {"status": "pending"}
        status = info["status"]
        lines.append(f"| {stage} | {manifest['jobs'].get(stage, {}).get('job_id', '')} | {status} |")
        if info.get("error"):
            lines.append(f"\n{stage}: `{info['error']}`\n")
    lines += ["", "| Stage / condition | Detection | False positives | Quality |",
              "| --- | ---: | ---: | ---: |"]
    for stage in stages:
        result = directory / stage / "results.json"
        if result.exists():
            data = json.loads(result.read_text())
            if stage == "latent_teacher":
                lines += ["", f"Autoregressive teacher validation passed: {data['passed']}; "
                          f"bit accuracy {data['bit_accuracy']:.1%}, AUROC {data['auroc']:.3f}.", ""]
            for name, metrics in data.get("conditions", {}).items():
                metrics = metrics.get("all", metrics)
                quality = (f"CLIP {metrics['clip_alignment']:.3f}" if "clip_alignment" in metrics
                           else f"human PPL {metrics['human_holdout_perplexity']:.2f}")
                lines.append(f"| {stage} / {name} | {metrics['detection']['rate']:.1%} | "
                             f"{metrics['false_positive']['rate']:.1%} | {quality} |")
    lines += ["", ("Owner detection uses the learned extractor and registered message. The payload is an identifier, "
              "not a cryptographic key. A train/test-separated text classifier probes outsider distinguishability. "
              "Teacher validation gates the long student run." if latent else
              "Image forgetting is tested separately on the fine-tuned class and other classes. "
              "JSON results retain confidence intervals and those domain-specific measurements. "
              "These are bounded experiments, not the paper's 50K-image FID reproduction."), ""]
    (directory / "report.md").write_text("\n".join(lines) + "\n")


def run(path, stage):
    manifest = json.loads(Path(path).read_text())
    directory = Path(manifest["directory"])
    job = os.environ.get("SLURM_JOB_ID", "local")
    receipt = directory / f"{stage}.json"
    with (directory / f"{stage}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt.exists() and json.loads(receipt.read_text())["status"] == "completed":
            return
        try:
            for dependency in STAGES[stage]:
                parent = json.loads((directory / f"{dependency}.json").read_text())
                if parent["status"] != "completed" or parent["job_id"] != manifest["jobs"][dependency]["job_id"]:
                    raise ValueError(f"Invalid completed dependency: {dependency}")
            for name, sha in manifest["code_sha256"].items():
                if file_hash(Path(manifest["source"]) / name) != sha:
                    raise ValueError("Frozen code changed")
            for name, sha in manifest.get("resource_code_sha256", {}).items():
                if file_hash(name) != sha:
                    raise ValueError("Official source or local model cards changed")
            if file_hash(manifest["data"]) != manifest["data_sha256"]:
                raise ValueError("Frozen data partitions changed")
            validate_partitions(json.loads(Path(manifest["data"]).read_text()))
            use_sources(manifest)
            seed_everything(manifest["seed"])
            write_json(receipt, {"status": "running", "job_id": job})
            with StopRequest() as stop:
                if stage.startswith("latent_"):
                    from ai_content_detector.watermarking import latent_text
                    if stage == "latent_preflight":
                        latent_text.preflight(manifest)
                    elif stage == "latent_embedder":
                        latent_text.train(manifest, stop)
                    elif stage == "latent_teacher":
                        latent_text.teacher_probe(manifest, stop)
                    elif stage == "latent_distill":
                        latent_text.fit(manifest, stop)
                    else:
                        latent_text.evaluate(manifest, stage, stop)
                elif stage == "preflight":
                    from ai_content_detector.watermarking.preflight import run as check
                    check(manifest)
                elif stage == "distill":
                    from ai_content_detector.watermarking.text import fit
                    fit(manifest, stop)
                elif stage in {"textseal", "distilled", "text_base", "text_clean"}:
                    from ai_content_detector.watermarking.text import evaluate
                    evaluate(manifest, stage, stop)
                else:
                    from ai_content_detector.watermarking.image import evaluate
                    evaluate(manifest, stage, stop)
            write_json(receipt, {"status": "completed", "job_id": job})
        except CheckpointPause:
            write_json(receipt, {"status": "paused", "job_id": job})
            subprocess.run(["scontrol", "requeue", job], check=True)
        except BaseException as error:
            write_json(receipt, {"status": "failed", "job_id": job, "error": repr(error)})
            raise
        finally:
            report(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["resources", "prepare", "adapt-text", "check", "submit", "report", "run"])
    parser.add_argument("manifest", nargs="?")
    parser.add_argument("stage", nargs="?", choices=STAGES)
    args = parser.parse_args()
    if args.action == "resources":
        print(resources())
    elif args.action == "prepare":
        print(prepare())
    elif args.action == "adapt-text":
        print(adapt_text(args.manifest))
    elif args.action == "submit":
        submit(args.manifest)
    elif args.action == "check":
        manifest = json.loads(Path(args.manifest).read_text())
        use_sources(manifest)
        from ai_content_detector.watermarking.preflight import offline
        offline(manifest)
    elif args.action == "report":
        report(args.manifest)
    else:
        run(args.manifest, args.stage)
