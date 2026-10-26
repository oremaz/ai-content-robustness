"""Scheme-agnostic attack workers. Owner evaluation runs in another process."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from ..rl_evasion.checkpointing import CheckpointPause, StopRequest, atomic_save
from ..rl_evasion.config import seed_everything
from .backbones import inject_lora, load_decoder, load_generator, state_for_generator, use_backbones
from .protocol import digest, file_hash, write_json


TASK_FIELDS = {"case", "method", "model", "clean_model", "fitting", "recipe", "output", "source",
               "backbones", "seed", "code_sha256"}
MODEL_FIELDS = {"architecture", "checkpoint", "decoder", "config"}
RECIPE_FIELDS = {"steps", "export_steps", "mode", "lr", "accumulation", "length", "batch", "lora_rank", "weights"}
PRIVATE_FIELDS = {"key", "message", "detector", "extractor", "watermark", "owner", "resources", "test", "scores"}


def validate_task(task):
    if set(task) - TASK_FIELDS or (TASK_FIELDS-{"clean_model"}) - set(task):
        raise ValueError("Unexpected attacker fields")
    if set(task["model"]) - MODEL_FIELDS or set(task["recipe"]) - RECIPE_FIELDS:
        raise ValueError("Unexpected model or recipe fields")
    if task["model"].get("architecture") not in {"causal_lm", "rar", "uvit"}:
        raise ValueError("Unsupported native architecture")
    def check(value):
        if isinstance(value, dict):
            if set(value) & PRIVATE_FIELDS:
                raise ValueError("Private evaluator fields in attacker input")
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(task)
    method = task["method"]
    if method == "merge":
        if set(task.get("clean_model", {})) != {"checkpoint"}:
            raise ValueError("Merging requires an explicit compatible clean checkpoint")
        weights = task["recipe"]["weights"]
        if not weights or len(set(weights)) != len(weights) or not all(0 <= w <= 1 for w in weights):
            raise ValueError("Invalid predetermined merge weights")
    elif method in {"clean_lora", "clean_full", "probe"}:
        if "clean_model" in task:
            raise ValueError("No-clean-weights worker cannot receive clean weights")
        if method != "probe" and task["recipe"]["mode"] != method.removeprefix("clean_"):
            raise ValueError("Training method and recipe disagree")
        steps = task["recipe"]["steps"]
        if steps < 1 or any(s < 1 or s > steps for s in task["recipe"]["export_steps"]):
            raise ValueError("Invalid fixed training budget")
    else:
        raise ValueError("Unsupported removal method")
    return digest(task)


def artifact_hashes(path):
    path = Path(path)
    return {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob("*"))
            if p.is_file() and not p.name.startswith(".")}


def verify_artifact(path, hashes):
    if not hashes or artifact_hashes(path) != hashes:
        raise ValueError("Frozen attack artifact changed")


def merge_state(marked, clean, weight):
    if marked.keys() != clean.keys() or any(marked[k].shape != clean[k].shape for k in marked):
        raise ValueError("Clean and supplied model states are incompatible")
    result = {}
    for key, tensor in marked.items():
        other = clean[key]
        if weight in (0., 1.):
            value = (other if weight == 0 else tensor).detach().cpu().to(tensor.dtype).clone()
        elif tensor.is_floating_point():
            value = other.detach().cpu().float().lerp(tensor.detach().cpu().float(), weight).to(tensor.dtype)
        else:
            if not torch.equal(tensor.cpu(), other.cpu()):
                raise ValueError(f"Incompatible nonfloating buffer: {key}")
            value = tensor.detach().cpu().clone()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"Invalid merged tensor: {key}")
        result[key] = value
    return result


def load_fitting(task):
    item = task["fitting"]
    if set(item) != {"path", "sha256"} or file_hash(item["path"]) != item["sha256"]:
        raise ValueError("Fitting data changed")
    data = json.loads(Path(item["path"]).read_text())
    if set(data) != {"train", "validation"} or not all(data.values()):
        raise ValueError("Attacker data must contain only training and validation")
    def identity(row):
        return row.get("id", row.get("path"))
    if set(map(identity, data["train"])) & set(map(identity, data["validation"])):
        raise ValueError("Fitting/validation leakage")
    return data


def image_bridge(task):
    data = load_fitting(task)
    output = Path(task["output"])
    data_path = output / "fitting.json"
    write_json(data_path, {"images": data})
    name = task["model"]["architecture"]
    family = "rar" if name == "rar" else "dcae"
    config = {**task["recipe"], "generic": True}
    return {"directory": str(output), "data": str(data_path), "image_training": config,
            "seed": task["seed"], "protocol_sha256": validate_task(task)}, family


def text_bridge(task):
    data = load_fitting(task)
    output = Path(task["output"])
    data_path = output / "fitting.json"
    write_json(data_path, {"text": {"removal": data["train"], "validation": data["validation"]}})
    config = task["recipe"]
    return {"directory": str(output), "data": str(data_path), "clean_source": task["model"]["checkpoint"],
            "clean_stage": "training", "text_training": {**config, "clean_steps": config["steps"],
                                                         "clean_lr": config["lr"]},
            "seed": task["seed"], "protocol_sha256": validate_task(task)}


def train_image(task, stop):
    from .image import fit_clean
    bridge, family = image_bridge(task)
    spec = task["model"]
    generator = load_generator(spec)
    decoder = load_decoder(spec)
    artifacts = []
    def save_candidate(name):
        step = int(name.rsplit("_", 1)[-1])
        folder = Path(task["output"]) / f"step-{step}"
        folder.mkdir(exist_ok=True)
        state = state_for_generator(generator, spec["architecture"])
        atomic_save({k: v.detach().cpu().clone() for k, v in state.items()}, folder / "model.pt")
        artifacts.append({"name": f"{task['method']}_{step}", "path": str(folder), "format": "image_state",
                          "mode": task["recipe"]["mode"], "step": step, "sha256": artifact_hashes(folder)})
    fit_clean(bridge, family, generator, decoder, stop, save_candidate)
    # Resumed training may have exported an earlier fixed checkpoint already.
    for step in task["recipe"]["export_steps"]:
        folder = Path(task["output"]) / f"step-{step}"
        if folder.exists() and not any(a["step"] == step for a in artifacts):
            artifacts.append({"name": f"{task['method']}_{step}", "path": str(folder), "format": "image_state",
                              "mode": task["recipe"]["mode"], "step": step, "sha256": artifact_hashes(folder)})
    return artifacts


def train_text(task, stop):
    from .text import fit
    fit(text_bridge(task), stop, clean=True)
    artifacts = []
    root = Path(task["output"]) / "training"
    for step in task["recipe"]["export_steps"]:
        final = step == task["recipe"]["steps"]
        folder = root / "model" if final else root / f"checkpoint-{step}"
        if not folder.exists():
            raise ValueError(f"Predetermined checkpoint missing: {step}")
        artifacts.append({"name": f"{task['method']}_{step}", "path": str(folder), "step": step,
                          "format": "hf_adapter" if not final and task["recipe"]["mode"] == "lora" else "hf",
                          "mode": task["recipe"]["mode"], "sha256": artifact_hashes(folder)})
    return artifacts


def merge(task, stop):
    spec = task["model"]
    text = spec["architecture"] == "causal_lm"
    if text:
        from .text import load_model
        model, tokenizer = load_model(spec["checkpoint"])
        model.cpu()
        clean, _ = load_model(task["clean_model"]["checkpoint"])
        clean.cpu()
        marked_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        clean_state = clean.state_dict()
    else:
        model = load_generator(spec, device="cpu")
        clean = load_generator(spec, task["clean_model"]["checkpoint"], device="cpu")
        marked_state = {k: v.detach().cpu().clone() for k, v in state_for_generator(model, spec["architecture"]).items()}
        clean_state = state_for_generator(clean, spec["architecture"])
    artifacts = []
    for weight in task["recipe"]["weights"]:
        folder = Path(task["output"]) / f"weight-{weight:g}"
        complete = folder / "complete.json"
        if not complete.exists():
            folder.mkdir(exist_ok=True)
            state = merge_state(marked_state, clean_state, weight)
            if text:
                model.load_state_dict(state, strict=True)
                model.save_pretrained(folder, max_shard_size="2GB")
                tokenizer.save_pretrained(folder)
            else:
                atomic_save(state, folder / "model.pt")
            write_json(complete, {"weight": weight})
            del state
        artifacts.append({"name": f"merge_{weight:g}", "path": str(folder), "format": "hf" if text else "image_state",
                          "mode": "full", "weight": weight, "sha256": artifact_hashes(folder)})
        if stop.requested:
            raise CheckpointPause("Fixed merge candidates saved")
    return artifacts


def probe(task):
    data, spec = load_fitting(task), task["model"]
    results = {}
    for mode in ("lora", "full"):
        torch.cuda.reset_peak_memory_stats()
        if spec["architecture"] == "causal_lm":
            from .text import load_model
            model, _ = load_model(spec["checkpoint"], training=True, training_mode=mode)
            ids = torch.tensor([data["train"][0]["tokens"][:task["recipe"]["length"]]], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        else:
            from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
            from PIL import Image
            model, decoder = load_generator(spec), load_decoder(spec)
            if mode == "full":
                model.requires_grad_(True)
            else:
                inject_lora(model, task["recipe"]["lora_rank"])
            model.train()
            row = data["train"][0]
            size = 256 if spec["architecture"] == "rar" else 512
            with Image.open(row["path"]) as image:
                image = Compose([Resize(size), CenterCrop(size), ToTensor()])(image.convert("RGB")).cuda()[None]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent = decoder.encode(image if size == 256 else image*2-1)
            labels = torch.tensor([row["label"]], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if spec["architecture"] == "rar":
                    from deps.rar.modeling.modules.losses import ARLoss
                    model.disable_kv_cache()
                    logits, target = model(latent, model.preprocess_condition(labels), return_labels=True)
                    loss, _ = ARLoss(model.config)(logits.float(), target)
                else:
                    loss, _ = model(latent*.2889, labels.int())
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-5, foreach=False)
        accumulation = task["recipe"]["accumulation"]
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(accumulation):
                if iteration or micro:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if spec["architecture"] == "causal_lm":
                            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
                        elif spec["architecture"] == "rar":
                            logits, target = model(latent, model.preprocess_condition(labels), return_labels=True)
                            loss, _ = ARLoss(model.config)(logits.float(), target)
                        else:
                            loss, _ = model(latent*.2889, labels.int())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Native clean training preflight failed")
                (loss / accumulation).backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm) or norm <= 0:
                raise FloatingPointError("Native clean training preflight failed")
            optimizer.step()
        results[mode] = {"loss": float(loss.detach()), "gradient_norm": float(norm),
                         "trainable_parameters": sum(p.numel() for p in parameters),
                         "optimizer_steps": 2, "accumulation": accumulation,
                         "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
        del parameters, optimizer, model, loss
        if spec["architecture"] != "causal_lm":
            del decoder, latent
            if spec["architecture"] == "rar":
                del logits, target
        gc.collect(); torch.cuda.empty_cache()
    return results


def run(path):
    task = json.loads(Path(path).read_text())
    signature = validate_task(task)
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "receipt.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["signature"] != signature:
            raise ValueError("Attack task changed")
        if previous["status"] == "completed":
            for artifact in previous.get("artifacts", []):
                verify_artifact(artifact["path"], artifact["sha256"])
            return
    for name, sha in task["code_sha256"].items():
        if file_hash(Path(task["source"]) / name) != sha:
            raise ValueError("Frozen attacker source changed")
    seal_path = Path(task["model"]["checkpoint"]).parent.parent / "input_seal.json"
    seal = json.loads(seal_path.read_text())
    verify_artifact(seal_path.parent / "supplied", seal["supplied"])
    if task["method"] == "merge":
        clean_seal = json.loads((seal_path.parent / "clean_seal.json").read_text())
        verify_artifact(seal_path.parent / "clean", clean_seal["clean"])
    use_backbones(task["backbones"])
    seed_everything(task["seed"])
    started = time.monotonic()
    write_json(receipt, {"status": "running", "signature": signature, "method": task["method"]})
    try:
        with StopRequest() as stop:
            if task["method"] == "probe":
                result, artifacts = probe(task), []
            elif task["method"] == "merge":
                result, artifacts = {}, merge(task, stop)
            elif task["model"]["architecture"] == "causal_lm":
                result, artifacts = {}, train_text(task, stop)
            else:
                result, artifacts = {}, train_image(task, stop)
        expected = len(task["recipe"]["weights"]) if task["method"] == "merge" else len(task["recipe"]["export_steps"])
        if task["method"] != "probe" and len(artifacts) != expected:
            raise ValueError("Predetermined candidates are incomplete")
        write_json(receipt, {"status": "completed", "signature": signature, "method": task["method"],
                            "artifacts": artifacts, "preflight": result, "elapsed_seconds": time.monotonic()-started,
                            "owner_detector_access": False, "clean_weights_access": task["method"] == "merge"})
    except CheckpointPause:
        write_json(receipt, {"status": "paused", "signature": signature})
        import os, subprocess
        subprocess.run(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]], check=True)
    except BaseException as error:
        write_json(receipt, {"status": "failed", "signature": signature, "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    run(parser.parse_args().task)
