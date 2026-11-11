"""Blind clean-data forgetting with a supplied-model prediction anchor."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from watermark_survival_pilot import ROOT, save, sha


def prepare(screen_manifest, evaluation_protocol=None, retrain_lora=False):
    import torch
    screen_manifest = Path(screen_manifest)
    screen = json.loads(screen_manifest.read_text())
    results = json.loads((Path(screen["directory"]) / "results.json").read_text())
    eligible = {name: checkpoint for name, checkpoint in screen["checkpoints"].items()
                if results["conditions"][name]["screen_passed"]}
    if set(eligible) != {"original", "fixed", "adapted"}:
        raise ValueError("All three initial checkpoints must pass before a matched attack comparison")
    if evaluation_protocol is not None:
        evaluation_protocol = Path(evaluation_protocol).resolve()
        protocol = json.loads(evaluation_protocol.read_text())
        if protocol["owner_threshold"] != screen["threshold"] or protocol["quality_rule"] != screen["quality_rule"]:
            raise ValueError("Evaluation threshold or quality rule differs from the initial screen")
        if protocol["validation"]["seed"] == protocol["final"]["seed"]:
            raise ValueError("Validation and final generation seeds must differ")
        if set(protocol["validation"]["labels"]) & set(protocol["final"]["labels"]):
            raise ValueError("Validation and final classes overlap")
        if any(len(protocol[stage]["labels"]) != protocol[stage]["n"] for stage in ["validation", "final"]):
            raise ValueError("Evaluation labels and sample counts disagree")
    parent = json.loads(Path(screen["parent_training"]).read_text())
    directory = ROOT / "results/research_workflow/paper_writing" / (
        ("preserving-attack-lora-" if retrain_lora else "preserving-attack-")
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    source = directory / "source"
    (source / "scripts").mkdir(parents=True)
    shutil.copytree(ROOT / "ai_content_detector", source / "ai_content_detector",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.ipynb", ".pytest_cache"))
    for name in ["watermark_preserving_attack.py", "watermark_survival_pilot.py", "watermark_transport_train.py",
                 "run_watermark_preserving_attack.slurm"]:
        shutil.copyfile(ROOT / "scripts" / name, source / "scripts" / name)
    fitting = json.loads(Path(parent["task"]["fitting"]["path"]).read_text())
    support = {r["path"] for r in fitting["train"][:2000]}
    attack_data = fitting["train"][4000:8000]
    if support & {r["path"] for r in attack_data}:
        raise ValueError("Attack fitting overlaps defense support")
    save(directory / "fitting.json", {"train": attack_data, "validation": fitting["validation"]})
    latents = torch.load(parent["latents"], map_location="cpu", weights_only=True)
    if len(latents) != len(fitting["train"]) or len(attack_data) != 4000:
        raise ValueError("Attack latent/fitting alignment changed")
    from ai_content_detector.rl_evasion.checkpointing import atomic_save
    atomic_save(latents[4000:8000].clone(), directory / "latents.pt")
    source_hashes = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    tasks = []
    for name, checkpoint in eligible.items():
        task = {"name": name, "directory": str(directory / name), "source": str(source),
                "model": {**screen["task"]["model"], "checkpoint": checkpoint},
                "backbones": screen["task"]["backbones"], "fitting": str(directory / "fitting.json"),
                "latents": str(directory / "latents.pt"), "seed": 2026110605,
                "recipe": {"steps": 500, "export_steps": [25, 100, 500], "batch": 4, "lr": 1.e-5,
                           "anchor_weights": [0., 1., 4., 16.], "gradient_clip": 1., "label_dropout": .1},
                "anchor": "Frozen copy of the supplied checkpoint, not original clean weights. Native denoising-prediction MSE on shared noisy clean latents and identical labels. No owner key or detector.",
                "source_hashes": source_hashes, "runtime_archive": screen["runtime_archive"],
                "compute": {"qos": "qos_gpu_h100-dev", "gpus": 1, "minutes": 30, "maximum_allocations": 2},
                "input_hashes": {str(p): sha(p) for p in [Path(checkpoint), directory / "latents.pt", directory / "fitting.json",
                                                         Path(screen["runtime_archive"]["path"])]}}
        if retrain_lora:
            task["recipe"].update(mode="lora", lora_rank=16, steps=2500,
                                  export_steps=[250, 1000, 2500], lr=1.e-4)
            task["compute"]["maximum_allocations"] = 4
        task["signature"] = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
        task_path = directory / (name + "-task.json")
        save(task_path, task)
        tasks.append(str(task_path))
    owner_manifest = {"directory": str(directory), "screen_manifest": str(screen_manifest),
                      "screen_results_sha256": sha(Path(screen["directory"]) / "results.json"), "tasks": tasks,
                      "scope": "Development removal study. Prediction anchoring is a quality-preservation regularizer, not a new watermark defense or an output-KL guarantee.",
                      "selection": "Evaluate paired preservation on fresh32validationseeds without owner scores. Choose largest passing update budget then smallest anchor weight; if none passes, retain the unmodified checkpoint and report that no nonzero attack meets the screen. Freeze selection before64freshfinalseeds. Retain every candidate and failure."}
    if evaluation_protocol is not None:
        target = directory/"evaluation_protocol.json"
        shutil.copyfile(evaluation_protocol,target)
        owner_manifest["evaluation_protocol"] = {"path":str(target),"sha256":sha(target)}
        owner_manifest["selection"] = protocol["selection"]
    save(directory / "manifest.json", owner_manifest)
    print(directory / "manifest.json")


def run(path):
    print("blind preserving attack starting", flush=True)
    import numpy as np
    import torch
    from ai_content_detector.rl_evasion.checkpointing import (
        CheckpointPause, StopRequest, atomic_save, rng_state, restore_rng, model_state, restore_model)
    from ai_content_detector.rl_evasion.config import seed_everything
    from ai_content_detector.watermarking.backbones import (
        use_backbones, load_generator, state_for_generator, inject_lora, merged_lora_state_for_generator)

    task = json.loads(Path(path).read_text())
    forbidden = {"key", "message", "detector", "extractor", "resources", "owner", "test", "scores"}
    def check(value):
        if isinstance(value, dict):
            if forbidden & set(value):raise ValueError("Private owner fields in blind attack")
            for child in value.values():check(child)
        elif isinstance(value, list):
            for child in value:check(child)
    check(task)
    signature = task.pop("signature")
    if hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest() != signature:
        raise ValueError("Blind attack task changed")
    task["signature"] = signature
    for filename, expected in task["input_hashes"].items():
        if sha(filename) != expected:raise ValueError("Blind attack input changed: " + filename)
    for relative, expected in task["source_hashes"].items():
        if sha(Path(task["source"]) / relative) != expected:raise ValueError("Frozen attack source changed")
    use_backbones(task["backbones"])
    data = json.loads(Path(task["fitting"]).read_text())
    latents = torch.load(task["latents"], map_location="cpu", weights_only=True)
    if len(latents) != len(data["train"]):raise ValueError("Blind attack fitting alignment changed")
    root = Path(task["directory"])
    root.mkdir(exist_ok=True)
    recipe = task["recipe"]
    reference = load_generator(task["model"])
    save(root / "environment.json", {"job_id": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__,
                                     "gpu": torch.cuda.get_device_name(), "python": sys.version})
    with StopRequest() as stop:
        for weight in recipe["anchor_weights"]:
            folder = root / f"anchor-{weight:g}"
            folder.mkdir(exist_ok=True)
            if (folder / "complete.json").exists():continue
            seed_everything(task["seed"])
            model = load_generator(task["model"])
            if recipe.get("mode") == "lora":
                inject_lora(model, recipe["lora_rank"])
            else:
                model.requires_grad_(True)
            parameters = [p for p in model.parameters() if p.requires_grad]
            if not parameters:
                raise ValueError("Attack has no trainable parameters")
            optimizer = torch.optim.AdamW(parameters, lr=recipe["lr"], weight_decay=0., foreach=False)
            restart = folder / "restart.pt"
            step, history = 0, []
            if restart.exists():
                state = torch.load(restart, map_location="cpu", weights_only=False)
                if state["signature"] != signature or state["anchor_weight"] != weight:raise ValueError("Attack restart changed")
                if state.get("model_format") == "trainable":
                    expected = {name for name, p in model.named_parameters() if p.requires_grad}
                    if set(state["model"]) != expected:
                        raise ValueError("Restart trainable parameters changed")
                    restore_model(model, state["model"])
                else:
                    model.load_state_dict(state["model"], strict=True)
                optimizer.load_state_dict(state["optimizer"])
                restore_rng(state["rng"])
                step, history = state["step"], state["history"]
                del state
            while step < recipe["steps"]:
                seed_everything(task["seed"] + step)
                indices = np.random.default_rng(task["seed"] + step).integers(len(latents), size=recipe["batch"])
                labels = torch.tensor([data["train"][i]["label"] for i in indices], device="cuda", dtype=torch.int32)
                labels[torch.rand(len(labels), device="cuda") < recipe["label_dropout"]] = 1000
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, info = model(latents[indices].cuda() * .2889, labels,
                                       forward_reference=reference.forward_without_cfg, reference_model_weight=weight)
                if not torch.isfinite(loss):raise FloatingPointError("Invalid preserving-attack loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, recipe["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if step % 25 == 0:
                    history.append({"step": step, "loss": float(loss.detach()),
                                    "anchor_mse": float(info["loss_dict"].get("loss_reference", 0.))})
                    print(task["name"], weight, history[-1], flush=True)
                if step in recipe["export_steps"]:
                    exported = (merged_lora_state_for_generator(model, "uvit") if recipe.get("mode") == "lora"
                                else {k: v.detach().cpu() for k, v in state_for_generator(model, "uvit").items()})
                    native = state_for_generator(reference, "uvit")
                    if exported.keys() != native.keys() or any(exported[k].shape != native[k].shape for k in native):
                        raise ValueError("Exported attack is not an ordinary native checkpoint")
                    if any(not torch.isfinite(v).all() for v in exported.values() if v.is_floating_point()):
                        raise FloatingPointError("Invalid exported attack checkpoint")
                    atomic_save(exported, folder / f"step-{step}.pt")
                    del exported, native
                if step % 50 == 0 or stop.requested or step == recipe["steps"]:
                    atomic_save({"signature": signature, "anchor_weight": weight, "step": step, "history": history,
                                 "model_format": "trainable" if recipe.get("mode") == "lora" else "full",
                                 "model": (model_state(model) if recipe.get("mode") == "lora" else
                                           {k: v.detach().cpu() for k, v in model.state_dict().items()}),
                                 "optimizer": optimizer.state_dict(), "rng": rng_state()}, restart)
                    save(folder / "progress.json", {"step": step, "history": history})
                if stop.requested:raise CheckpointPause("Blind preserving-attack restart saved")
            save(folder / "complete.json", {"signature": signature, "anchor_weight": weight,
                 "exports": {str(step): sha(folder / f"step-{step}.pt") for step in recipe["export_steps"]},
                 "owner_feedback_used": False, "original_clean_weights_used": False})
            del model, optimizer, parameters
            torch.cuda.empty_cache()
        save(root / "complete.json", {"signature": signature, "owner_feedback_used": False,
                                      "status": "Blind attacks frozen. Preservation selection and owner evaluation are separate."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("manifest")
    parser.add_argument("--evaluation-protocol")
    parser.add_argument("--retrain-lora", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":prepare(args.manifest,args.evaluation_protocol,args.retrain_lora)
    else:
        import signal
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
        try:run(args.manifest)
        except CheckpointPause as error:
            print(str(error), flush=True)
            sys.exit(75)
