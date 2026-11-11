"""Initial paired generation screen for completed meta-distillation checkpoints."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from watermark_survival_pilot import ROOT, save, sha


def prepare(parent, seed=2026110604):
    parent = Path(parent)
    config = json.loads(parent.read_text())
    training = Path(config["directory"])
    checkpoints = {"original": config["task"]["model"]["checkpoint"]}
    for variant in ["fixed", "adapted"]:
        receipt = json.loads((training / variant / "results.json").read_text())
        filename = training / variant / "model.pt"
        if receipt["signature"] != config["signature"] or sha(filename) != receipt["model_sha256"]:
            raise ValueError("Training result differs from checkpoint")
        checkpoints[variant] = str(filename)
    directory = ROOT / "results/research_workflow/paper_writing" / (
        "transport-screen-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    source = directory / "source"
    (source / "scripts").mkdir(parents=True)
    shutil.copytree(ROOT / "ai_content_detector", source / "ai_content_detector",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.ipynb", ".pytest_cache"))
    for name in ["watermark_transport_screen.py", "watermark_survival_pilot.py", "watermark_transport_train.py",
                 "run_watermark_transport_screen.slurm"]:
        shutil.copyfile(ROOT / "scripts" / name, source / "scripts" / name)
    fitting = json.loads(Path(config["task"]["fitting"]["path"]).read_text())["train"]
    classes = sorted(set(range(1000)) - {r["label"] for r in fitting})
    classes.sort(key=lambda label: hashlib.sha256(f"{seed}:{label}".encode()).hexdigest())
    config.update(directory=str(directory), source=str(source), parent_training=str(parent),
                  checkpoints=checkpoints, seed=seed, samples=32, batch=4, labels=classes[:32],
                  compute={"qos": "qos_gpu_h100-dev", "gpus": 1, "minutes": 30, "maximum_allocations": 2},
                  status="Initial generation screen only. No removal robustness claim.")
    threshold_path = Path(config["parent"]).parent / "dcae/results.json"
    config["threshold"] = json.loads(threshold_path.read_text())["threshold"]
    files = [parent, threshold_path, Path(config["task"]["model"]["decoder"]),
             Path(config["resources"]["cards"]) / "detector_dc-ae.yaml",
             Path(config["runtime_archive"]["path"])]
    files += [Path(p) for p in checkpoints.values()]
    files += [Path(config["resources"]["files"][name]["path"]) for name in
              ["detector_dcae_diffusion.pt", "detector_dcae_diffusion_msg.npy"]]
    files += [p for p in Path(config["clip"]["path"]).rglob("*") if p.is_file()]
    config["input_hashes"] = {str(p): sha(p) for p in files}
    config["source_hashes"] = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    config.pop("signature", None)
    config["signature"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    save(directory / "manifest.json", config)
    print(directory / "manifest.json")


def run(path):
    print("checkpoint screen starting", flush=True)
    import numpy as np
    from PIL import Image
    import torch
    from unittest.mock import patch
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
    from ai_content_detector.rl_evasion.config import seed_everything
    from ai_content_detector.watermarking.backbones import use_backbones, load_generator, load_decoder
    from ai_content_detector.watermarking.image import bit_scores, cards
    from ai_content_detector.watermarking.image_quality import ImageQuality
    from ai_content_detector.watermarking.preservation import numeric_gate

    config = json.loads(Path(path).read_text())
    root = Path(config["directory"])
    for filename, expected in config["input_hashes"].items():
        if sha(filename) != expected:
            raise ValueError("Screen input changed: " + filename)
    for relative, expected in config["source_hashes"].items():
        if sha(Path(config["source"]) / relative) != expected:
            raise ValueError("Screen source changed: " + relative)
    use_backbones(config["task"]["backbones"])
    sys.path.insert(0, config["resources"]["repositories"]["distseal"]["path"])
    with patch("importlib.metadata.packages_distributions", return_value=config["package_map"]):
        from distseal.loader import load_detector
    decoder = load_decoder(config["task"]["model"])
    detector, message = load_detector(cards(config, "dcae")[2], device="cuda")
    detector.eval().requires_grad_(False)
    save(root / "environment.json", {"job_id": os.environ.get("SLURM_JOB_ID"), "python": sys.version,
                                     "torch": torch.__version__, "gpu": torch.cuda.get_device_name()})
    with StopRequest() as stop:
        for variant, checkpoint in config["checkpoints"].items():
            model = load_generator(config["task"]["model"], checkpoint=checkpoint)
            folder = root / variant
            folder.mkdir(exist_ok=True)
            output = folder / "samples.json"
            rows = json.loads(output.read_text())["rows"] if output.exists() else []
            if [r["index"] for r in rows] != list(range(len(rows))):
                raise ValueError("Screen restart indices changed")
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
                    raise CheckpointPause("Checkpoint-screen image batch saved")
            del model
            torch.cuda.empty_cache()
        del decoder, detector
        torch.cuda.empty_cache()
        metrics = ImageQuality(config["clip"]["path"], config["metric_cache"])
        threshold = config["threshold"]
        summaries = {}
        for variant in config["checkpoints"]:
            folder = root / variant
            output = folder / "quality.json"
            rows = json.loads(output.read_text()) if output.exists() else []
            samples = json.loads((folder / "samples.json").read_text())["rows"]
            for start in range(len(rows), config["samples"], config["batch"]):
                indices = list(range(start, min(start+config["batch"], config["samples"])))
                originals = [Image.open(root / "original" / f"{i:05}.png").convert("RGB") for i in indices]
                candidates = [Image.open(folder / f"{i:05}.png").convert("RGB") for i in indices]
                for index, value in zip(indices, metrics.score(originals, candidates, [config["labels"][i] for i in indices])):
                    reference = value["reference_clip_alignment"]
                    value["class_alignment_retention"] = value["clip_alignment"] / reference if reference > 0 else None
                    rows.append({"index": index, **value, "preservation": numeric_gate(value, config["quality_rule"])})
                save(output, rows)
                if stop.requested:
                    raise CheckpointPause("Checkpoint-screen quality batch saved")
            summary = {"n": len(samples), "detected": sum(r["bit_accuracy"] > threshold for r in samples),
                       "preserved": sum(r["preservation"]["passed"] for r in rows),
                       "mean_lpips": float(np.mean([r["lpips"] for r in rows])),
                       "mean_bit_accuracy": float(np.mean([r["bit_accuracy"] for r in samples]))}
            summary["screen_passed"] = summary["detected"] >= 28 and summary["preserved"] >= 28
            summaries[variant] = summary
        save(root / "results.json", {"threshold": threshold, "conditions": summaries, "status": config["status"]})
        print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("manifest")
    parser.add_argument("--seed", type=int, default=2026110604)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.manifest, args.seed)
    else:
        from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
        try:
            run(args.manifest)
        except CheckpointPause as error:
            print(str(error), flush=True)
            sys.exit(75)
