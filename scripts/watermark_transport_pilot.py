"""Validate a marked adapted-reference teacher before robustness training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from watermark_survival_pilot import ROOT, dependency_map, save, sha


def prepare(adaptation_steps=100, evaluation_seed=2026110602):
    previous = ROOT / "results/research_workflow/paper_writing/survival-pilot-20261106T170840Z/manifest.json"
    config = json.loads(previous.read_text())
    directory = ROOT / "results/research_workflow/paper_writing" / (
        "transport-teacher-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    source = directory / "source"
    (source / "scripts").mkdir(parents=True)
    shutil.copytree(ROOT / "ai_content_detector", source / "ai_content_detector",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.ipynb", ".pytest_cache"))
    for name in ("watermark_transport_pilot.py", "watermark_survival_pilot.py",
                 "run_watermark_transport_pilot.slurm"):
        shutil.copyfile(ROOT / "scripts" / name, source / "scripts" / name)
    config.update(directory=str(directory), source=str(source), seed=2026110602, evaluation_seed=evaluation_seed,
                  adaptation={"steps": adaptation_steps, "lr": 1.e-5, "batch": 4,
                              "optimizer": "AdamW", "weight_decay": 0.,
                              "checkpoint_every": max(10, adaptation_steps//10),
                              "fit_indices": [0, 2000], "attack_indices": [4000, 8000]},
                  samples=32, batch=4, variants=["fixed", "adapted"],
                  continue_rule="Each marked teacher must retain at least28/32 owner detections and28/32 preservation passes against its own clean reference. No defense or novelty claim follows from this screen.",
                  status="Development teacher validation, not a robustness experiment.",
                  scope="Native DistSeal latent embedder with its released0.5 strength; compare W(Gclean) with W(A(Gclean)). Adaptation uses only clean latents and labels; no owner detector enters adaptation.",
                  compute={"qos": "qos_gpu_h100-dev", "gpus": 1, "minutes": 30, "maximum_allocations": 2},
                  package_map=dependency_map())
    fitting = json.loads(Path(config["task"]["fitting"]["path"]).read_text())
    classes = sorted(set(range(1000)) - {row["label"] for row in fitting["train"]})
    classes.sort(key=lambda label: hashlib.sha256(f"{evaluation_seed}:{label}".encode()).hexdigest())
    config["labels"] = classes[:config["samples"]]
    weights = config["resources"]["files"]
    config["input_hashes"].update({weights[name]["path"]: weights[name]["sha256"] for name in
                                  ["detector_dcae_diffusion.pt", "detector_dcae_diffusion_msg.npy"]})
    config["input_hashes"][str(previous)] = sha(previous)
    config["source_hashes"] = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    config.pop("signature", None)
    config["signature"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    save(directory / "manifest.json", config)
    print(directory / "manifest.json")


def run(path):
    print("teacher worker starting", flush=True)
    import numpy as np
    from PIL import Image
    import torch
    from unittest.mock import patch
    from ai_content_detector.rl_evasion.checkpointing import (
        CheckpointPause, StopRequest, atomic_save, rng_state, restore_rng)
    from ai_content_detector.rl_evasion.config import seed_everything
    from ai_content_detector.watermarking.backbones import use_backbones, load_generator, load_decoder
    from ai_content_detector.watermarking.image import bit_scores, cards
    from ai_content_detector.watermarking.image_quality import ImageQuality
    from ai_content_detector.watermarking.preservation import numeric_gate

    config = json.loads(Path(path).read_text())
    root = Path(config["directory"])
    for filename, expected in config["input_hashes"].items():
        if sha(filename) != expected:
            raise ValueError("Teacher input changed: " + filename)
    for relative, expected in config["source_hashes"].items():
        if sha(Path(config["source"]) / relative) != expected:
            raise ValueError("Teacher source changed: " + relative)
    print("teacher inputs and source verified", flush=True)
    use_backbones(config["task"]["backbones"])
    sys.path.insert(0, config["resources"]["repositories"]["distseal"]["path"])
    with patch("importlib.metadata.packages_distributions", return_value=config["package_map"]):
        from distseal.loader import load_watermarker
    print("native teacher imports complete", flush=True)
    save(root / "environment.json", {"job_id": os.environ.get("SLURM_JOB_ID"), "python": sys.version,
                                     "torch": torch.__version__, "gpu": torch.cuda.get_device_name()})
    model = load_generator(config["task"]["model"], checkpoint=config["clean_checkpoint"])
    decoder = load_decoder(config["task"]["model"])
    watermarker, message, strength = load_watermarker(cards(config, "dcae")[2], device="cuda")
    if strength != .5:
        raise ValueError("Unexpected released teacher strength")
    watermarker.eval().requires_grad_(False)
    watermarker.blender.scaling_w = strength
    detector = watermarker.detector
    fitting = json.loads(Path(config["task"]["fitting"]["path"]).read_text())["train"]
    latents = torch.load(config["latents"], map_location="cpu", weights_only=True)
    if len(latents) != len(fitting):
        raise ValueError("Teacher fitting data are not aligned")
    with StopRequest() as stop:
        for variant in config["variants"]:
            folder = root / variant
            folder.mkdir(exist_ok=True)
            if variant == "adapted":
                model.requires_grad_(True)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["adaptation"]["lr"],
                                              weight_decay=0., foreach=False)
                restart = root / "adaptation.pt"
                step, history = 0, []
                if restart.exists():
                    state = torch.load(restart, map_location="cpu", weights_only=False)
                    if state["signature"] != config["signature"]:
                        raise ValueError("Teacher adaptation restart changed")
                    model.load_state_dict(state["model"], strict=True)
                    optimizer.load_state_dict(state["optimizer"])
                    restore_rng(state["rng"])
                    step, history = state["step"], state["history"]
                    del state
                while step < config["adaptation"]["steps"]:
                    seed_everything(config["seed"] + step)
                    indices = torch.randperm(2000)[:config["adaptation"]["batch"]]
                    labels = torch.tensor([fitting[i]["label"] for i in indices], device="cuda", dtype=torch.int32)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, _ = model(latents[indices].cuda() * .2889, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite teacher clean loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    step += 1
                    history.append(float(loss.detach()))
                    if step % config["adaptation"].get("checkpoint_every", 10) == 0 or stop.requested:
                        atomic_save({"signature": config["signature"], "step": step, "history": history,
                                     "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                     "optimizer": optimizer.state_dict(), "rng": rng_state()}, restart)
                        print("adaptation", step, history[-1], flush=True)
                    if stop.requested:
                        raise CheckpointPause("Teacher adaptation saved")
                model.zero_grad(set_to_none=True)
                model.requires_grad_(False)
                del optimizer
                torch.cuda.empty_cache()
            sample_path = folder / "samples.json"
            rows = json.loads(sample_path.read_text())["rows"] if sample_path.exists() else []
            if [row["index"] for row in rows] != list(range(len(rows))):
                raise ValueError("Teacher generation restart indices differ")
            for start in range(len(rows), config["samples"], config["batch"]):
                seed_everything(config.get("evaluation_seed",config["seed"]) + 10000 + start)
                labels = torch.tensor(config["labels"][start:start+config["batch"]], device="cuda", dtype=torch.int32)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = model.generate(labels, torch.full_like(labels, 1000), scale=1.) / .2889
                    payload = message.reshape(1, -1).expand(len(labels), -1)
                    marked = watermarker.blender(raw, watermarker.embedder(raw, payload))
                    decoded = decoder.decode(torch.cat([raw, marked])).float().mul(127.5).add(128).clamp(0, 255).to(torch.uint8)
                    clean_pixels, marked_pixels = decoded.chunk(2)
                    clean_scores = bit_scores(detector, message, clean_pixels.float() / 255.)
                    marked_scores = bit_scores(detector, message, marked_pixels.float() / 255.)
                for offset, (clean_image, marked_image, clean_score, marked_score) in enumerate(
                        zip(clean_pixels, marked_pixels, clean_scores, marked_scores)):
                    index = start + offset
                    record = {"index": index, "label": config["labels"][index],
                              "clean_bit_accuracy": clean_score, "marked_bit_accuracy": marked_score}
                    for name, image in [("clean", clean_image), ("marked", marked_image)]:
                        filename = folder / f"{index:05}-{name}.png"
                        Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(filename)
                        record[name + "_sha256"] = sha(filename)
                    rows.append(record)
                save(sample_path, {"signature": config["signature"], "rows": rows})
                print(variant, "samples", len(rows), flush=True)
                if stop.requested:
                    raise CheckpointPause("Teacher generation saved")
        del model, decoder, watermarker, detector, latents
        torch.cuda.empty_cache()
        metrics = ImageQuality(config["clip"]["path"], config["metric_cache"])
        threshold = json.loads((Path(config["parent"]).parent / "dcae/results.json").read_text())["threshold"]
        summaries = {}
        for variant in config["variants"]:
            folder = root / variant
            quality_path = folder / "quality.json"
            rows = json.loads(quality_path.read_text()) if quality_path.exists() else []
            samples = json.loads((folder / "samples.json").read_text())["rows"]
            for start in range(len(rows), config["samples"], config["batch"]):
                indices = list(range(start, min(start+config["batch"], config["samples"])))
                originals = [Image.open(folder / f"{i:05}-clean.png").convert("RGB") for i in indices]
                candidates = [Image.open(folder / f"{i:05}-marked.png").convert("RGB") for i in indices]
                for index, value in zip(indices, metrics.score(originals, candidates, [config["labels"][i] for i in indices])):
                    reference = value["reference_clip_alignment"]
                    value["class_alignment_retention"] = value["clip_alignment"] / reference if reference > 0 else None
                    rows.append({"index": index, **value, "preservation": numeric_gate(value, config["quality_rule"])})
                save(quality_path, rows)
                if stop.requested:
                    raise CheckpointPause("Teacher quality saved")
            summaries[variant] = {"n": len(samples),
                "clean_detected": sum(r["clean_bit_accuracy"] > threshold for r in samples),
                "marked_detected": sum(r["marked_bit_accuracy"] > threshold for r in samples),
                "preserved": sum(r["preservation"]["passed"] for r in rows),
                "mean_lpips": float(np.mean([r["lpips"] for r in rows]))}
        passed = all(r["marked_detected"] >= 28 and r["preserved"] >= 28 for r in summaries.values())
        save(root / "results.json", {"threshold": threshold, "conditions": summaries, "teacher_screen_passed": passed,
                                      "status": config["status"]})
        print(json.dumps(summaries), "screen_passed", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("manifest", nargs="?")
    parser.add_argument("--adaptation-steps", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=2026110602)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.adaptation_steps < 1 or args.evaluation_seed < 0:
            parser.error("Adaptation steps must be positive and evaluation seed nonnegative")
        prepare(args.adaptation_steps,args.evaluation_seed)
    else:
        from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
        try:
            run(args.manifest)
        except CheckpointPause as error:
            print(str(error), flush=True)
            sys.exit(75)
