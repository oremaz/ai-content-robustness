"""Checkpointed official-DDA evaluation and one SD1.5 attacker/defender round."""

from __future__ import annotations

import ast
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ...detectors.published_image import DDADetector
from ..checkpointing import (CheckpointClock, CheckpointPause, atomic_save, gradient_state,
                             load_checkpoint, model_state, restore_gradients, restore_model,
                             restore_rng, rng_state)
from ..config import ImageEvasionConfig, seed_everything


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def detector_for(manifest, checkpoint=None, race=False, device="cuda"):
    resource = manifest["resources"]
    return DDADetector(checkpoint=checkpoint or resource["checkpoint"]["files"]["DDA_ckpt.pth"]["path"],
                       dinov2_source=resource["dinov2"]["path"],
                       dda_source=resource["Dual-Data-Alignment"]["path"], device=device,
                       input_size=200 if race else None)


def notebook_helpers(path):
    from torchvision import datasets, transforms
    notebook = json.loads(Path(path).read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "def build_binary_image_dataset(" in "".join(cell.get("source", [])))
    nodes = ast.parse(source).body
    end = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Tuple) and any(isinstance(item, ast.Name) and item.id == "train_ds"
                       for item in target.elts) for target in node.targets))
    namespace = {"Path": Path, "np": np, "torch": torch, "datasets": datasets, "transforms": transforms}
    exec(compile(ast.Module(nodes[:end], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def partition_rows(samples, seed=42):
    from sklearn.model_selection import train_test_split
    paths = [str(path) for path, _ in samples]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate image paths in ArtiFact")
    labels = np.asarray([target for _, target in samples])
    selected, _ = train_test_split(np.arange(len(samples)), train_size=0.15, stratify=labels, random_state=seed)
    train, evaluation = train_test_split(selected, test_size=0.2, stratify=labels[selected], random_state=seed)
    def rows(indices):
        return [{"path": paths[index], "label": 1 - int(labels[index])} for index in indices]
    return rows(train), rows(evaluation)


def prepare_data(manifest, stop):
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    root, repo = Path(manifest["directory"]), Path(manifest["repository"])
    destination = root / "partitions.json"
    if destination.exists():
        return
    helpers = notebook_helpers(manifest["notebook"])
    dataset = helpers["build_binary_image_dataset"](repo / "data/artifact-real-fake")
    train, evaluation = partition_rows(dataset.samples)
    artifact = evaluation
    external = {}
    for name, location in (("CIFAKE", repo / "data/cifake/test"), ("Shoes", repo / "data/shoes-ai-vs-real")):
        split = helpers["find_imagefolder_split"](location)
        folder = helpers["datasets"].ImageFolder(split)
        mapping = {index: 0 if label.lower() == "real" else 1 if label.lower() in {"fake", "ai", "ai-midjourney"} else None
                   for index, label in enumerate(folder.classes)}
        if None in mapping.values():
            raise ValueError(f"Unrecognized {name} labels: {folder.classes}")
        external[name] = [{"path": str(path), "label": mapping[label]} for path, label in folder.samples]
    tokenizer = AutoTokenizer.from_pretrained(manifest["models"]["sd"]["path"], subfolder="tokenizer", local_files_only=True)
    table = pq.read_table(Path(manifest["models"]["prompts"]["path"]) / "data/train.parquet")
    column = next(name for name in ("Prompt", "prompt", "text") if name in table.column_names)
    candidates = table[column].to_pylist()
    prompts, seen = [], set()
    for index in np.random.default_rng(42).permutation(len(candidates)):
        prompt = candidates[index]
        tokens = tuple(tokenizer(prompt, truncation=True, max_length=tokenizer.model_max_length)["input_ids"])
        if tokens in seen:
            continue
        seen.add(tokens)
        prompts.append({"prompt": prompt, "source_index": int(index), "token_ids": list(tokens),
                        "sample_id": digest(list(tokens)), "seed": 42 + len(prompts)})
        if len(prompts) == 1500:
            break
    if len(prompts) != 1500:
        raise ValueError("Insufficient unique SD-tokenized prompts")
    seen_images = set()
    def unique_real(pool, count):
        result = []
        for row in pool:
            if row["label"] != 0:
                continue
            sha256 = file_hash(row["path"])
            if sha256 in seen_images:
                continue
            seen_images.add(sha256)
            result.append({**row, "sha256": sha256})
            if len(result) == count:
                return result
        raise ValueError("Insufficient content-disjoint ArtiFact real images")
    real_train = unique_real(train, 1000)
    real_eval = unique_real(evaluation, 500)
    payload = {"artifact_holdout": artifact, "external": external,
               "fitting": prompts[:1000], "evaluation": prompts[1000:],
               "real_fitting": real_train, "real_evaluation": real_eval,
               "source_counts": {"artifact_total": len(dataset), "artifact_train_pool": len(train),
                                 "artifact_holdout": len(artifact)},
               "seed": 42, "label_convention": {"real": 0, "fake": 1},
               "notebook_sha256": file_hash(manifest["notebook"])}
    write_json(destination, payload)
    print("Data partitions saved", payload["source_counts"], flush=True)
    if stop.requested:
        raise CheckpointPause("Immutable image partitions saved")


class Images(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            values = self.transform(image)
        return values, float(row["label"])


def score_images(detector, rows, destination, stop, batch_size=16):
    for row in rows:
        if "sha256" in row and file_hash(row["path"]) != row["sha256"]:
            raise ValueError("An immutable evaluation image has changed")
    signature = {"rows": digest(rows), "checkpoint": file_hash(detector._paths()[0]),
                 "input_size": detector.input_size, "batch_size": batch_size}
    state = load_checkpoint(destination, signature) or {"config": signature, "scores": []}
    dataset = Images(rows, detector.transform)
    detector._load()
    loader = DataLoader(dataset, batch_size=batch_size, sampler=list(range(len(state["scores"]), len(rows))),
                        num_workers=4, pin_memory=True)
    for values, _ in loader:
        if stop.requested:
            atomic_save(state, destination)
            raise CheckpointPause("Image score progress saved")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=detector.device.startswith("cuda")):
            scores = detector._model(values.to(detector.device, non_blocking=True)).float().sigmoid().cpu().tolist()
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite image classifier scores")
        state["scores"].extend(scores)
        atomic_save(state, destination)
    return np.asarray(state["scores"], dtype=float)


def metrics(rows, scores, threshold=0.5):
    from ..defender_comparison.protocol import classification_metrics
    from ..text_evasion.evaluate import compute_tpr_at_fpr
    from sklearn.metrics import average_precision_score, balanced_accuracy_score
    labels = np.asarray([row["label"] for row in rows])
    scores = np.asarray(scores)
    if scores.shape != labels.shape or not np.isfinite(scores).all() or set(labels) != {0, 1}:
        raise ValueError("Image metrics require aligned, finite scores and both classes")
    return {**classification_metrics(scores[labels == 1], scores[labels == 0], threshold),
            "balanced_accuracy": float(balanced_accuracy_score(labels, scores >= threshold)),
            "average_precision_ai": float(average_precision_score(labels, scores)),
            "tpr_at_1pct_fpr": compute_tpr_at_fpr(scores, labels, 0.01),
            "tpr_at_5pct_fpr": compute_tpr_at_fpr(scores, labels, 0.05)}


def benchmark(manifest, stop):
    root = Path(manifest["directory"])
    partitions = json.loads((root / "partitions.json").read_text())
    detector = detector_for(manifest)
    records = {}
    for name, rows in {"ArtiFact holdout": partitions["artifact_holdout"], **partitions["external"]}.items():
        scores = score_images(detector, rows, root / "benchmark" / f"{name}.pt", stop)
        records[name] = metrics(rows, scores)
        write_json(root / "benchmark/measurements.json", records)
        print(name, records[name], flush=True)


def sd_pipeline(manifest):
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    pipeline = StableDiffusionPipeline.from_pretrained(manifest["models"]["sd"]["path"],
                                                       torch_dtype=torch.bfloat16, local_files_only=True,
                                                       safety_checker=None, requires_safety_checker=False).to("cuda")
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def generate_images(pipeline, prompts, directory, identity, stop):
    directory = Path(directory)
    signature = {"prompts": digest(prompts), "model": identity, "steps": 50, "eta": 1.0, "guidance": 7.5}
    record = directory / "images.json"
    payload = json.loads(record.read_text()) if record.exists() else {"signature": signature, "rows": []}
    if payload["signature"] != signature:
        raise ValueError("Saved image generation inputs changed")
    directory.mkdir(parents=True, exist_ok=True)
    for row in payload["rows"]:
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("A saved generated image has changed")
    for start in range(len(payload["rows"]), len(prompts), 8):
        if stop.requested:
            raise CheckpointPause("Image generation manifest retained")
        batch = prompts[start:start + 8]
        generators = [torch.Generator(device="cuda").manual_seed(row["seed"]) for row in batch]
        with torch.inference_mode():
            images = pipeline([row["prompt"] for row in batch], num_inference_steps=50,
                              guidance_scale=7.5, eta=1.0, generator=generators).images
        for offset, (image, row) in enumerate(zip(images, batch)):
            path = directory / f"{start + offset:05d}.png"
            temporary = path.with_suffix(".tmp.png")
            image.save(temporary)
            temporary.replace(path)
            payload["rows"].append({**row, "path": str(path), "label": 1, "sha256": file_hash(path)})
        write_json(record, payload)
        print("Generation", directory.name, len(payload["rows"]), "/", len(prompts), flush=True)
    return payload["rows"]


def split_defender_rows(fake, real):
    if len(fake) != len(real) or len(fake) < 5:
        raise ValueError("Defender data requires aligned real and generated pools")
    cut = int(0.8 * len(fake))
    train, validation = fake[:cut] + real[:cut], fake[cut:] + real[cut:]
    if {row["path"] for row in train} & {row["path"] for row in validation}:
        raise ValueError("Defender training and validation images overlap")
    return train, validation


def fit_defender(manifest, initial_checkpoint, fake, real, destination, stop):
    destination = Path(destination)
    final = destination / "completed.pt"
    train, validation = split_defender_rows(fake, real)
    signature = {"train": digest(train), "validation": digest(validation),
                 "initial_checkpoint": str(initial_checkpoint), "initial_sha256": file_hash(initial_checkpoint),
                 "epochs": 3, "batch_size": 16, "accumulation": 4, "learning_rate": 2e-5, "seed": 42}
    if final.exists():
        load_checkpoint(final, signature)
        return str(final)
    seed_everything(42)
    detector = detector_for(manifest, initial_checkpoint, race=True)
    detector._load()
    model = detector._model
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=2e-5)
    restart_path = destination / "restart.pt"
    state = load_checkpoint(restart_path, signature)
    progress = {"epoch": 0, "batch": 0, "best_loss": float("inf"), "history": []}
    if state:
        restore_model(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_gradients(model, state["gradients"])
        restore_rng(state["rng"])
        progress = state["progress"]
    clock = CheckpointClock()
    def save():
        atomic_save({"config": signature, "model": model_state(model), "optimizer": optimizer.state_dict(),
                     "gradients": gradient_state(model), "rng": rng_state(), "progress": progress}, restart_path)
        clock.saved()
    save()
    train_data, val_data = Images(train, detector.transform), Images(validation, detector.transform)
    for epoch in range(progress["epoch"], 3):
        order = np.random.default_rng(42 + epoch).permutation(len(train)).tolist()
        total_batches = (len(order) + 15) // 16
        loader = DataLoader(train_data, batch_size=16, sampler=order[progress["batch"] * 16:],
                            num_workers=4, pin_memory=True)
        model.train()
        for batch_number, (values, labels) in enumerate(loader, progress["batch"]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(values.to("cuda", non_blocking=True)).float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.to("cuda").float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite DDA adaptation loss")
            accumulation = min(4, total_batches - (batch_number // 4) * 4)
            (loss / accumulation).backward()
            if (batch_number + 1) % 4 == 0 or batch_number + 1 == total_batches:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                               error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad()
            progress["batch"] = batch_number + 1
            if clock.due() or stop.requested:
                save()
                if stop.requested:
                    raise CheckpointPause("DDA adaptation optimizer and gradients saved")
        model.eval()
        losses = []
        for values, labels in DataLoader(val_data, batch_size=16, num_workers=4):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(values.to("cuda")).float()
                losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.to("cuda").float(), reduction="sum")))
        val_loss = sum(losses) / len(validation)
        if not np.isfinite(val_loss):
            raise FloatingPointError("Nonfinite DDA validation loss")
        progress["history"].append({"epoch": epoch + 1, "validation_loss": val_loss})
        if val_loss < progress["best_loss"]:
            progress["best_loss"] = val_loss
            atomic_save({"config": signature, "model": model_state(model),
                         "base_checkpoint": manifest["resources"]["checkpoint"]["files"]["DDA_ckpt.pth"]["path"]},
                        destination / "best.pt")
        progress.update(epoch=epoch + 1, batch=0)
        save()
        write_json(destination / "history.json", progress["history"])
        print("DDA validation", destination.name, epoch + 1, val_loss, flush=True)
        if stop.requested:
            raise CheckpointPause("DDA adaptation epoch saved")
    best = load_checkpoint(destination / "best.pt", signature)
    atomic_save(best, final)
    del optimizer, model, detector
    gc.collect()
    torch.cuda.empty_cache()
    return str(final)


def phase_evaluation(manifest, checkpoint, fake, real, name, stop, threshold=None):
    root = Path(manifest["directory"])
    detector = detector_for(manifest, checkpoint, race=True)
    rows = fake + real
    scores = score_images(detector, rows, root / "scoring" / f"{name}.pt", stop)
    result = {"fixed_0.5": metrics(rows, scores), "evaluation_ids": [row.get("sample_id", row["path"]) for row in rows],
              "image_identities": digest(rows), "checkpoint_sha256": file_hash(checkpoint)}
    if threshold is not None:
        result["warmup_validation_threshold"] = metrics(rows, scores, threshold)
    write_json(root / "measurements" / f"{name}.json", result)
    detector.unload()
    return result


def warmup(manifest, stop):
    root = Path(manifest["directory"])
    data = json.loads((root / "partitions.json").read_text())
    pipeline = sd_pipeline(manifest)
    fake_fit = generate_images(pipeline, data["fitting"], root / "base_fitting", manifest["models"]["sd"]["revision"], stop)
    fake_eval = generate_images(pipeline, data["evaluation"], root / "base_evaluation", manifest["models"]["sd"]["revision"], stop)
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    initial = manifest["resources"]["checkpoint"]["files"]["DDA_ckpt.pth"]["path"]
    phase_evaluation(manifest, initial, fake_eval, data["real_evaluation"], "official_on_base", stop)
    checkpoint = fit_defender(manifest, initial, fake_fit, data["real_fitting"], root / "warmup", stop)
    from ..defender_comparison.protocol import validation_threshold
    detector = detector_for(manifest, checkpoint, race=True)
    validation_real = data["real_fitting"][800:]
    scores = score_images(detector, validation_real, root / "warmup/validation_real_scores.pt", stop)
    threshold = validation_threshold(scores)
    write_json(root / "warmup/threshold.json", {"threshold": threshold, "human_validation_samples": len(scores)})
    detector.unload()
    phase_evaluation(manifest, checkpoint, fake_eval, data["real_evaluation"], "warmup_on_base", stop, threshold)


def attack(manifest, stop):
    from dataclasses import asdict, replace
    root = Path(manifest["directory"])
    data = json.loads((root / "partitions.json").read_text())
    config = ImageEvasionConfig(**manifest["attacker_config"])
    warm_identity = json.loads((root / "measurements/warmup_on_base.json").read_text())["checkpoint_sha256"]
    if file_hash(root / "warmup/completed.pt") != warm_identity:
        raise ValueError("Warmed reward detector changed after its baseline evaluation")
    signature = {"attacker_config": asdict(config), "warmup_sha256": warm_identity,
                 "fitting_prompts": digest(data["fitting"][:800])}
    calibration_path = root / "reference_reward_calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        if calibration["signature"] != signature:
            raise ValueError("Reference reward calibration inputs changed")
    else:
        probe_config = replace(config, output_dir=str(root / "reference_probe"), num_train_epochs=1,
                               reward_reference_weight=0.0, measure_updated_reference_kl=True)
        probe = image_trainer(manifest, probe_config, data)
        probe.train()
        raw = probe.training_history[-1]["updated_initial_kl_per_sample"]
        scale = select_reference_scale(raw)
        indices = np.random.default_rng(config.seed).permutation(800)[:config.sample_batch_size]
        calibration = {"signature": signature, "scale": scale, "method": "median positive per-image KL after one unanchored probe epoch",
                       "probe_sample_ids": [data["fitting"][int(index)]["sample_id"] for index in indices],
                       "raw_kl_per_sample": raw, "probe_history": probe.training_history,
                       "probe_weights_used_for_main_training": False,
                       "reward_at_scale": 0.5, "reference_weight": config.reward_reference_weight}
        write_json(calibration_path, calibration)
        del probe
        gc.collect()
        torch.cuda.empty_cache()
    config = replace(config, reference_kl_scale=calibration["scale"])
    print("Frozen reference reward KL scale:", calibration["scale"], flush=True)
    resolved_path = root / "effective_attacker_config.json"
    if resolved_path.exists() and json.loads(resolved_path.read_text()) != asdict(config):
        raise ValueError("Effective attacker configuration changed")
    write_json(resolved_path, asdict(config))
    if stop.requested:
        raise CheckpointPause("Reference reward calibration and effective configuration saved")
    write_json(root / "attacker/reward_identity.json", {"sha256": warm_identity, "race_resize": 200,
                                                       "calibration_sha256": file_hash(calibration_path)})
    trainer = image_trainer(manifest, config, data)
    trainer.train()
    write_json(root / "attacker/training_history.json", trainer.training_history)


def select_reference_scale(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Reference calibration requires finite, nonnegative KL samples")
    positive = values[values > 0]
    if not positive.size:
        raise ValueError("Probe did not produce measurable KL drift; cannot calibrate reference reward")
    return float(np.median(positive))


def image_trainer(manifest, config, data):
    from .ddpo_trainer import CompositeImageReward, DDPOImageEvasionTrainer, ImageDetectorReward
    root = Path(manifest["directory"])
    seed_everything(config.seed)
    trainer = DDPOImageEvasionTrainer(config)
    trainer._setup_model()
    trainer.pipeline.safety_checker = None
    trainer.prompts = [row["prompt"] for row in data["fitting"][:800]]
    trainer.eval_prompts = [row["prompt"] for row in data["evaluation"]]
    detector = detector_for(manifest, root / "warmup/completed.pt", race=True)
    trainer.reward_fn = CompositeImageReward([ImageDetectorReward(detector, name="dda")], clip_model_id=config.clip_reward_model, weights={
        "evasion": config.reward_evasion_weight, "clip": config.reward_clip_weight, "aesthetic": 0.0})
    return trainer


def defend(manifest, stop):
    root = Path(manifest["directory"])
    data = json.loads((root / "partitions.json").read_text())
    if manifest.get("attacker_parent_manifest"):
        from .ablation import defender_inputs
        fake_fit, fake_eval, base = defender_inputs(manifest, data, stop)
    else:
        pipeline = sd_pipeline(manifest)
        effective = json.loads((root / "effective_attacker_config.json").read_text())
        expected = dict(manifest["attacker_config"])
        expected["reference_kl_scale"] = json.loads((root / "reference_reward_calibration.json").read_text())["scale"]
        if effective != expected:
            raise ValueError("Effective attacker configuration differs from its recorded reward calibration")
        restart = load_checkpoint(root / "attacker/restart.pt", effective)
        if restart is None or restart["progress"]["epoch"] != manifest["attacker_config"]["num_train_epochs"]:
            raise ValueError("DDPO attacker is not complete")
        if (restart["prompts"] != [row["prompt"] for row in data["fitting"][:800]]
                or restart["eval_prompts"] != [row["prompt"] for row in data["evaluation"]]):
            raise ValueError("Attacker checkpoint prompts differ from the recorded partitions")
        from peft import LoraConfig
        pipeline.unet.requires_grad_(False)
        pipeline.unet.add_adapter(LoraConfig(r=8, lora_alpha=16, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        restore_model(pipeline.unet, restart["model"])
        pipeline.unet.eval()
        identity = file_hash(root / "attacker/restart.pt")
        fake_fit = generate_images(pipeline, data["fitting"], root / "attacked_fitting", identity, stop)
        fake_eval = generate_images(pipeline, data["evaluation"], root / "attacked_evaluation", identity, stop)
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        base = json.loads((root / "base_evaluation/images.json").read_text())["rows"]
    threshold = json.loads((root / "warmup/threshold.json").read_text())["threshold"]
    warmed = root / "warmup/completed.pt"
    before_base = phase_evaluation(manifest, warmed, base, data["real_evaluation"], "warmup_on_base", stop, threshold)
    before_attack = phase_evaluation(manifest, warmed, fake_eval, data["real_evaluation"], "warmup_on_attacked", stop, threshold)
    updated = fit_defender(manifest, warmed, fake_fit, data["real_fitting"], root / "updated", stop)
    after_attack = phase_evaluation(manifest, updated, fake_eval, data["real_evaluation"], "updated_on_attacked", stop, threshold)
    after_base = phase_evaluation(manifest, updated, base, data["real_evaluation"], "updated_on_base", stop, threshold)
    if (before_base["image_identities"] != after_base["image_identities"]
            or before_attack["image_identities"] != after_attack["image_identities"]):
        raise ValueError("Before/after defender evaluation images differ")
    write_json(root / "results.json", {"rounds": 1, "warmup": str(warmed), "updated": updated,
                                       "attacker_epochs": manifest["attacker_config"]["num_train_epochs"],
                                       "eval_prompts": len(data["evaluation"]), "saved_images_reused": True})


def prepare(manifest, stop):
    from dataclasses import replace
    import shutil
    from .ddpo_trainer import CompositeImageReward, DDPOImageEvasionTrainer, ImageDetectorReward
    prepare_data(manifest, stop)
    root = Path(manifest["directory"])
    if (root / "preflight.json").exists():
        return
    data = json.loads((root / "partitions.json").read_text())
    detector = detector_for(manifest, race=True)
    with Image.open(data["real_evaluation"][0]["path"]) as image:
        score = float(detector.predict_batch([image])[0])
    config = replace(ImageEvasionConfig(**manifest["attacker_config"]), output_dir=str(root / "preflight_attacker"),
                     num_train_epochs=1, num_inference_steps=2, sample_batch_size=2, train_batch_size=1)
    trainer = DDPOImageEvasionTrainer(config)
    trainer._setup_model()
    trainer.prompts = [row["prompt"] for row in data["fitting"][:2]]
    trainer.eval_prompts = [data["evaluation"][0]["prompt"]]
    trainer.reward_fn = CompositeImageReward([ImageDetectorReward(detector, name="dda")], clip_model_id=config.clip_reward_model, weights={
        "evasion": 0.64, "clip": 0.26, "aesthetic": 0.0})
    trainer.train()
    research_summary = {"device": torch.cuda.get_device_name(), "official_dda_real_score": score,
                        "ddpo_history": trainer.training_history,
                        "covers": ["DDA forward", "SD sampling", "CLIP reward", "initial-reference KL", "LoRA backward and optimizer"]}
    write_json(root / "preflight.json", research_summary)
    del trainer
    detector.unload()
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(root / "preflight_attacker")
