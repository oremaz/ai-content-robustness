"""Development pilot for post-distillation watermark survival filtering."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]


def dependency_map():
    result = {}
    for distribution in importlib.metadata.distributions():
        declared = (distribution.read_text("top_level.txt") or "").split()
        if not declared:
            records = csv.reader((distribution.read_text("RECORD") or "").splitlines())
            declared = {row[0].split("/")[0].removesuffix(".py") for row in records if row}
        for name in declared:
            if name.isidentifier():
                result.setdefault(name, []).append(distribution.metadata["Name"])
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare():
    parent = ROOT / "results/research_workflow/watermarking/20261026T001727478096Z/manifest.json"
    removal = ROOT / "results/research_workflow/watermark_removal/20261026T110140724565Z"
    task_path = removal / "tasks/model03/clean_full.json"
    task = json.loads(task_path.read_text())
    prior = json.loads(parent.read_text())
    directory = ROOT / "results/research_workflow/paper_writing" / ("survival-pilot-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    source = directory / "source"
    (source / "scripts").mkdir(parents=True)
    shutil.copytree(ROOT / "ai_content_detector", source / "ai_content_detector",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.ipynb", ".pytest_cache"))
    for name in ("watermark_survival_pilot.py", "run_watermark_survival_pilot.slurm"):
        shutil.copyfile(ROOT / "scripts" / name, source / "scripts" / name)
    fitting = json.loads(Path(task["fitting"]["path"]).read_text())
    classes = sorted(set(range(1000)) - {row["label"] for row in fitting["train"]})
    if len(classes) != 200:
        raise ValueError("Expected the original 800/200 fitting-class partition")
    latents = Path(task["output"]) / "dcae_clean/fitting_latents.pt"
    clean = json.loads((removal / "manifest.json").read_text())["cases"]["model03"]["clean"]
    inputs = [parent, task_path, Path(task["fitting"]["path"]), latents,
              Path(task["model"]["checkpoint"]), Path(clean), Path(task["model"]["decoder"])]
    quality_config = json.loads((ROOT / "results/research_workflow/image_quality/20261104T153348508006Z/manifest.json").read_text())
    classes.sort(key=lambda label: hashlib.sha256(f"2026110601:{label}".encode()).hexdigest())
    config = {"directory": str(directory), "source": str(source), "task": task,
              "clean_checkpoint": clean, "latents": str(latents), "parent": str(parent),
              "seed": 2026110601, "curvature_examples": 32, "samples": 32, "batch": 4,
              "labels": classes[:32], "variants": ["original", "inverse", "survival1", "survival4"],
              "resources": prior["resources"], "clip": prior["clip"],
              "metric_cache": quality_config["metric_cache"],
              "status": "Development pilot only; no confirmatory inference or established novelty claim.",
              "quality_rule": {"lpips": [0., .2], "ssim": [.6, 1.000001],
                               "clip_image_cosine": [.9, 1.000001], "class_alignment_retention": [.95, None]},
              "continue_rule": "Do not launch removal attacks unless a candidate retains at least 28/32 detections and 28/32 paired preservation passes. This is a development screen, not a statistical guarantee.",
              "scope": "Block-constant mean squared gradients of native clean loss approximate curvature. The survival filter is a heuristic; no Hessian or global robustness guarantee is claimed. Original watermark and extractor are fixed. This pilot does not yet harden TextSeal2 or test adversarial preservation.",
              "package_map": dependency_map(),
              "compute": {"qos": "qos_gpu_h100-dev", "gpus": 1, "minutes": 30, "maximum_allocations": 2},
              "input_hashes": {str(p): sha(p) for p in inputs},
              "source_hashes": {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}}
    config["signature"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    save(directory / "manifest.json", config)
    print(directory / "manifest.json")


def run(path):
    import numpy as np
    from PIL import Image
    import torch
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
    from ai_content_detector.rl_evasion.config import seed_everything
    from ai_content_detector.watermarking.backbones import use_backbones, load_generator, load_decoder
    from ai_content_detector.watermarking.image import bit_scores, cards
    from ai_content_detector.watermarking.image_quality import ImageQuality
    from ai_content_detector.watermarking.preservation import numeric_gate
    from ai_content_detector.watermarking.survival import survival_gains

    config = json.loads(Path(path).read_text())
    root = Path(config["directory"])
    for source, expected in config["input_hashes"].items():
        if sha(source) != expected:
            raise ValueError("Pilot input changed: " + source)
    for relative, expected in config["source_hashes"].items():
        if sha(Path(config["source"]) / relative) != expected:
            raise ValueError("Frozen pilot source changed: " + relative)
    use_backbones(config["task"]["backbones"])
    sys.path.insert(0, config["resources"]["repositories"]["distseal"]["path"])
    from unittest.mock import patch
    # Diffusers' module map otherwise stats every installed file on Lustre.
    with patch("importlib.metadata.packages_distributions", return_value=config["package_map"]):
        from distseal.loader import load_detector
    save(root / "environment.json", {"job_id": os.environ.get("SLURM_JOB_ID"), "python": sys.version,
         "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
         "packages": {name: importlib.metadata.version(name) for name in ["transformers", "numpy", "torchvision"]}})
    spec = config["task"]["model"]
    model = load_generator(spec, checkpoint=config["clean_checkpoint"]).requires_grad_(True)
    clean = {k: p.detach().cpu().clone() for k, p in model.named_parameters()}
    names = set(clean)
    fitting = json.loads(Path(config["task"]["fitting"]["path"]).read_text())["train"]
    latents = torch.load(config["latents"], map_location="cpu", weights_only=True)
    if len(latents) != len(fitting):
        raise ValueError("Cached fitting latent count changed")
    progress_path = root / "curvature.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "signature": config["signature"], "step": 0, "curvature": dict.fromkeys(sorted(names), 0.), "losses": []}
    if progress["signature"] != config["signature"]:
        raise ValueError("Curvature restart protocol changed")
    with StopRequest() as stop:
        for step in range(progress["step"], config["curvature_examples"]):
            seed_everything(config["seed"] + step)
            model.zero_grad(set_to_none=True)
            label = torch.tensor([fitting[step]["label"]], device="cuda", dtype=torch.int32)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(latents[step:step+1].cuda() * .2889, label)
            loss.backward()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite clean pilot loss")
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    value = float(parameter.grad.float().square().mean())
                    if not np.isfinite(value):
                        raise FloatingPointError("Nonfinite curvature proxy")
                    progress["curvature"][name] += value / config["curvature_examples"]
            progress["step"] = step + 1
            progress["losses"].append(float(loss.detach()))
            save(progress_path, progress)
            print("curvature", step + 1, float(loss.detach()), flush=True)
            if stop.requested:
                raise CheckpointPause("Curvature pass saved")
        model.zero_grad(set_to_none=True)
        del model, latents
        torch.cuda.empty_cache()
        model = load_generator(spec)
        marked = {k: p.detach().cpu().clone() for k, p in model.named_parameters()}
        if set(marked) != names:
            raise ValueError("Clean and marked parameter names differ")
        delta = {k: marked[k].float() - clean[k].float() for k in names}
        energy = {k: float(v.double().square().sum()) for k, v in delta.items()}
        decoder = load_decoder(spec)
        detector, message = load_detector(cards(config, "dcae")[2], device="cuda")
        detector.eval().requires_grad_(False)
        summaries = {}
        for variant in config["variants"]:
            folder = root / variant
            folder.mkdir(exist_ok=True)
            if variant == "original":
                gains, receipt = dict.fromkeys(names, 1.), {"control": "unchanged released watermark"}
            else:
                horizon = {"inverse": 0., "survival1": 1., "survival4": 4.}[variant]
                gains, receipt = survival_gains(progress["curvature"], energy, horizon)
            save(folder / "filter.json", {"gains": gains, "receipt": receipt})
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_((clean[name] + gains[name] * delta[name]).to(parameter.device))
            output = folder / "samples.json"
            rows = json.loads(output.read_text())["rows"] if output.exists() else []
            if [r["index"] for r in rows] != list(range(len(rows))):
                raise ValueError("Pilot generation restart indices differ")
            for start in range(len(rows), config["samples"], config["batch"]):
                seed_everything(config["seed"] + 10000 + start)
                labels = torch.tensor(config["labels"][start:start+config["batch"]], device="cuda", dtype=torch.int32)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model.generate(labels, torch.full_like(labels, 1000), scale=1.)
                    pixels = decoder.decode(z / .2889).float().mul(127.5).add(128).clamp(0, 255).to(torch.uint8)
                    scores = bit_scores(detector, message, pixels.float() / 255.)
                for offset, (image, score) in enumerate(zip(pixels, scores)):
                    index = start + offset
                    filename = folder / f"{index:05}.png"
                    Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(filename)
                    rows.append({"index": index, "label": config["labels"][index], "bit_accuracy": score,
                                 "sha256": sha(filename)})
                save(output, {"signature": config["signature"], "rows": rows})
                print(variant, "samples", len(rows), flush=True)
                if stop.requested:
                    raise CheckpointPause("Pilot generation batch saved")
            summaries[variant] = {"mean_bit_accuracy": float(np.mean([r["bit_accuracy"] for r in rows]))}
        del model, decoder, detector
        torch.cuda.empty_cache()
        metrics = ImageQuality(config["clip"]["path"], config["metric_cache"])
        threshold = json.loads((Path(config["parent"]).parent / "dcae/results.json").read_text())["threshold"]
        for variant in config["variants"]:
            folder = root / variant
            output = folder / "quality.json"
            rows = json.loads(output.read_text()) if output.exists() else []
            samples = json.loads((folder / "samples.json").read_text())["rows"]
            for start in range(len(rows), config["samples"], config["batch"]):
                indices = range(start, min(start+config["batch"], config["samples"]))
                originals = [Image.open(root / "original" / f"{i:05}.png").convert("RGB") for i in indices]
                candidates = [Image.open(folder / f"{i:05}.png").convert("RGB") for i in indices]
                values = metrics.score(originals, candidates, config["labels"][start:start+config["batch"]])
                for index, value in zip(indices, values):
                    reference = value["reference_clip_alignment"]
                    value["class_alignment_retention"] = value["clip_alignment"] / reference if reference > 0 else None
                    rows.append({"index": index, **value, "preservation": numeric_gate(value, config["quality_rule"])})
                save(output, rows)
                if stop.requested:
                    raise CheckpointPause("Pilot quality batch saved")
            summaries[variant].update(detected=sum(r["bit_accuracy"] > threshold for r in samples),
                                      preservation_pass=sum(r["preservation"]["passed"] for r in rows), n=len(rows))
        save(root / "results.json", {"status": config["status"], "threshold": threshold, "conditions": summaries})
        print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("manifest", nargs="?")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
        try:
            run(args.manifest)
        except CheckpointPause as error:
            print(str(error), flush=True)
            sys.exit(75)
