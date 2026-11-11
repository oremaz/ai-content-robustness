"""Matched fixed-reference and adapted-reference watermark meta-distillation pilots."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from watermark_survival_pilot import ROOT, dependency_map, save, sha


def prepare(teacher_manifest, retention_kind="denoising", retention_weight=1.):
    previous = Path(teacher_manifest)
    config = json.loads(previous.read_text())
    result_path = Path(config["directory"]) / "results.json"
    result = json.loads(result_path.read_text())
    if not result["teacher_screen_passed"]:
        raise ValueError("Both teachers must pass the frozen quality/detection screen")
    directory = ROOT / "results/research_workflow/paper_writing" / (
        "transport-train-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    source = directory / "source"
    (source / "scripts").mkdir(parents=True)
    shutil.copytree(ROOT / "ai_content_detector", source / "ai_content_detector",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.ipynb", ".pytest_cache"))
    for name in ("watermark_transport_train.py", "watermark_survival_pilot.py", "run_watermark_transport_train.slurm"):
        shutil.copyfile(ROOT / "scripts" / name, source / "scripts" / name)
    config.update(directory=str(directory), source=str(source), teacher_manifest=str(previous),
                  adapted_teacher=str(Path(config["directory"]) / "adaptation.pt"),
                  training={"outer_steps": 100, "inner_steps": config["adaptation"]["steps"], "inner_lr": 1.e-5,
                            "outer_lr": 2.e-6, "batch": 4, "target_examples": 256,
                            "retention_kind": retention_kind, "retention_weight": retention_weight, "gradient_clip": 1.,
                            "approximation": "First-order MAML, detached fresh AdamW inner optimizer; identity Jacobian. Both variants use exactly the teacher's frozen support sequence and noise seeds. No exact Hessian or Adam meta-gradient is claimed."},
                  compute={"qos": "qos_gpu_h100-dev", "gpus_per_variant": 1, "minutes": 30,
                           "maximum_allocations_per_variant": 8, "variants": ["fixed", "adapted"]},
                  status="Matched development experiment. Does not establish novelty or confirmatory robustness.",
                  package_map=dependency_map())
    archive = ROOT / "results/research_workflow/paper_writing/resources/native_python_packages.json"
    config["runtime_archive"] = json.loads(archive.read_text())
    config["input_hashes"].update({str(previous): sha(previous), str(result_path): sha(result_path),
                                   config["runtime_archive"]["path"]: config["runtime_archive"]["sha256"],
                                   config["adapted_teacher"]: sha(config["adapted_teacher"])})
    config["source_hashes"] = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    config.pop("signature", None)
    config["signature"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    save(directory / "manifest.json", config)
    print(directory / "manifest.json")


def bootstrap(manifest, destination):
    import zipfile
    archive = json.loads(Path(manifest).read_text())["runtime_archive"]
    if sha(archive["path"]) != archive["sha256"]:
        raise ValueError("Installed-package mirror changed")
    with zipfile.ZipFile(archive["path"]) as contents:
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in contents.namelist()):
            raise ValueError("Unsafe package mirror entry")
        contents.extractall(destination)


def run(path, variant):
    print("meta-distillation worker starting", variant, flush=True)
    import torch
    from unittest.mock import patch
    from ai_content_detector.rl_evasion.checkpointing import (
        CheckpointPause, StopRequest, atomic_save, rng_state, restore_rng)
    from ai_content_detector.rl_evasion.config import seed_everything
    from ai_content_detector.watermarking.backbones import use_backbones, load_generator
    from ai_content_detector.watermarking.image import cards

    config = json.loads(Path(path).read_text())
    root = Path(config["directory"]) / variant
    root.mkdir(exist_ok=True)
    for filename, expected in config["input_hashes"].items():
        if sha(filename) != expected:
            raise ValueError("Training input changed: " + filename)
    for relative, expected in config["source_hashes"].items():
        if sha(Path(config["source"]) / relative) != expected:
            raise ValueError("Training source changed: " + relative)
    print("meta-distillation inputs verified", flush=True)
    use_backbones(config["task"]["backbones"])
    sys.path.insert(0, config["resources"]["repositories"]["distseal"]["path"])
    with patch("importlib.metadata.packages_distributions", return_value=config["package_map"]):
        from distseal.loader import load_watermarker
    save(root / "environment.json", {"job_id": os.environ.get("SLURM_JOB_ID"), "python": sys.version,
                                     "torch": torch.__version__, "gpu": torch.cuda.get_device_name()})
    recipe = config["training"]
    spec = config["task"]["model"]
    fitting = json.loads(Path(config["task"]["fitting"]["path"]).read_text())["train"]
    latents = torch.load(config["latents"], map_location="cpu", weights_only=True)
    if len(latents) != len(fitting) or len(fitting) != 8000:
        raise ValueError("Unexpected clean fitting alignment")
    if {r["path"] for r in fitting[:2000]} & {r["path"] for r in fitting[4000:]}:
        raise ValueError("Defense-support / future-attack overlap")
    target_path = root / "targets.pt"
    with StopRequest() as stop:
        if not target_path.exists():
            teacher = load_generator(spec, checkpoint=config["clean_checkpoint"])
            watermarker, message, strength = load_watermarker(cards(config, "dcae")[2], device="cuda")
            watermarker.eval().requires_grad_(False)
            if strength != .5:
                raise ValueError("Unexpected native watermark strength")
            watermarker.blender.scaling_w = strength
            seed_everything(config["seed"] + 500000)
            labels = torch.randint(0, 800, (recipe["target_examples"],)).tolist()
            # The original fitting class set is not necessarily the integers0..799.
            classes = sorted({r["label"] for r in fitting})
            labels = [classes[i] for i in labels]
            target_parts = {}
            for name in ["fixed", "adapted"]:
                if name == "adapted":
                    checkpoint = torch.load(config["adapted_teacher"], map_location="cpu", weights_only=False, mmap=True)
                    if checkpoint["step"] != recipe["inner_steps"]:
                        raise ValueError("Adapted teacher and student inner horizons differ")
                    teacher.load_state_dict(checkpoint["model"], strict=True)
                    del checkpoint
                parts = []
                for start in range(0, len(labels), recipe["batch"]):
                    chunk_path = root / f"targets-{name}-{start:05}.pt"
                    if chunk_path.exists():
                        part = torch.load(chunk_path, map_location="cpu", weights_only=True)
                    else:
                        seed_everything(config["seed"] + 600000 + start)
                        batch_labels = torch.tensor(labels[start:start+recipe["batch"]], device="cuda", dtype=torch.int32)
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            raw = teacher.generate(batch_labels, torch.full_like(batch_labels, 1000), scale=1.) / .2889
                            payload = message.reshape(1, -1).expand(len(batch_labels), -1)
                            marked = watermarker.blender(raw, watermarker.embedder(raw, payload)) * .2889
                        part = marked.float().cpu()
                        atomic_save(part, chunk_path)
                    parts.append(part)
                    print("targets", name, start + len(part), flush=True)
                    if stop.requested:
                        raise CheckpointPause("Marked teacher target chunk saved")
                target_parts[name] = torch.cat(parts)
            atomic_save({"signature": config["signature"], "labels": labels, **target_parts}, target_path)
            del teacher, watermarker, message, target_parts, parts, part, raw, marked
            torch.cuda.empty_cache()
        targets = torch.load(target_path, map_location="cpu", weights_only=True)
        if targets["signature"] != config["signature"]:
            raise ValueError("Teacher targets changed")
        model = load_generator(spec).requires_grad_(True)
        reference = load_generator(spec) if recipe.get("retention_kind", "denoising") == "prediction" else None
        parameters = list(model.parameters())
        outer_optimizer = torch.optim.AdamW(parameters, lr=recipe["outer_lr"], weight_decay=0., foreach=False)
        restart = root / "restart.pt"
        step, history = 0, []
        if restart.exists():
            state = torch.load(restart, map_location="cpu", weights_only=False)
            if state["signature"] != config["signature"] or state["variant"] != variant:
                raise ValueError("Meta-distillation restart changed")
            model.load_state_dict(state["model"], strict=True)
            outer_optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"])
            step, history = state["step"], state["history"]
            del state
        backup = [p.detach().clone() for p in parameters]
        while step < recipe["outer_steps"]:
            with torch.no_grad():
                for saved, parameter in zip(backup, parameters):
                    saved.copy_(parameter)
            inner_optimizer = torch.optim.AdamW(parameters, lr=recipe["inner_lr"], weight_decay=0., foreach=False)
            for inner in range(recipe["inner_steps"]):
                seed_everything(config["seed"] + inner)
                indices = torch.randperm(2000)[:recipe["batch"]]
                labels = torch.tensor([fitting[i]["label"] for i in indices], device="cuda", dtype=torch.int32)
                inner_optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = model(latents[indices].cuda() * .2889, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, recipe["gradient_clip"], error_if_nonfinite=True)
                inner_optimizer.step()
                if stop.requested:
                    break
            del inner_optimizer
            model.zero_grad(set_to_none=True)
            if not stop.requested:
                seed_everything(config["seed"] + 700000 + step)
                indices = torch.randperm(recipe["target_examples"])[:recipe["batch"]]
                labels = torch.tensor([targets["labels"][i] for i in indices], device="cuda", dtype=torch.int32)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    query, _ = model(targets[variant][indices].cuda(), labels)
                query.backward()
            with torch.no_grad():
                for parameter, saved in zip(parameters, backup):
                    parameter.copy_(saved)
            if not stop.requested:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if reference is None:
                        retention, _ = model(targets["fixed"][indices].cuda(), labels)
                    else:
                        timesteps, _, noisy = model.train_scheduler.sample(targets["fixed"][indices].cuda())
                        with torch.no_grad():
                            anchor = reference.forward_without_cfg(noisy, timesteps, labels)
                        prediction = model.forward_without_cfg(noisy, timesteps, labels)
                        retention = torch.nn.functional.mse_loss(prediction.float(), anchor.float())
                    weighted_retention = recipe["retention_weight"] * retention
                # Query gradients stay attached to the restored parameter objects.
                weighted_retention.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, recipe["gradient_clip"], error_if_nonfinite=True)
                outer_optimizer.step()
                step += 1
                history.append({"step": step, "query": float(query.detach()),
                                "retention": float(retention.detach()), "gradient_norm": float(norm)})
                print(variant, history[-1], flush=True)
            model.zero_grad(set_to_none=True)
            if step % 5 == 0 or stop.requested or step == recipe["outer_steps"]:
                atomic_save({"signature": config["signature"], "variant": variant, "step": step,
                             "history": history, "rng": rng_state(), "optimizer": outer_optimizer.state_dict(),
                             "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, restart)
                save(root / "progress.json", {"step": step, "history": history, "variant": variant})
            if stop.requested:
                raise CheckpointPause("Meta-distillation saved at outer boundary; at most one unfinished inner branch is repeated")
        # Native UViT loading expects the ModuleDict's "uvit." prefix.
        exported = {k: v.detach().cpu() for k, v in model.get_trainable_modules().state_dict().items()}
        atomic_save(exported, root / "model.pt")
        save(root / "results.json", {"signature": config["signature"], "variant": variant,
                                    "steps": step, "model_sha256": sha(root / "model.pt"),
                                    "status": "Training completed; fresh preservation-qualified removal evaluation still required."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "bootstrap"])
    parser.add_argument("manifest")
    parser.add_argument("variant", nargs="?", choices=["fixed", "adapted"])
    parser.add_argument("--destination")
    parser.add_argument("--retention-kind", choices=["denoising", "prediction"], default="denoising")
    parser.add_argument("--retention-weight", type=float, default=1.)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.retention_weight <= 0:
            parser.error("Retention weight must be positive")
        prepare(args.manifest, args.retention_kind, args.retention_weight)
    elif args.action == "bootstrap":
        bootstrap(args.manifest, args.destination)
    else:
        import signal
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
        try:
            run(args.manifest, args.variant)
        except CheckpointPause as error:
            print(str(error), flush=True)
            sys.exit(75)
