"""Training-only attacker ablations with validation selection and frozen test data."""

from __future__ import annotations

from dataclasses import asdict, replace
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import research
from .ddpo_trainer import (ImageDetectorReward, decode_latents_to_pil, ddim_step_with_logprob,
                           pipeline_sample_with_logprob, previous_ddim_timestep)
from ..checkpointing import (CheckpointPause, atomic_save, load_checkpoint, model_state,
                             restore_model, restore_rng, rng_state)
from ..config import ImageEvasionConfig


def pilot_variants():
    common = {"num_train_epochs": 100, "diagnostic_logging": True, "measure_reference_kl": True,
              "validation_interval": 20, "adapter_save_interval": 10000}
    large = {"rollout_batches": 4, "num_train_epochs": 25, "validation_interval": 5}
    grouped = {"samples_per_prompt": 4, "advantage_mode": "group"}
    combined = {**large, **grouped, "evasion_mode": "scaled_logit"}
    return {
        "control": {**common},
        "larger_rollout": {**common, **large},
        "reliable_baseline": {**common, "prompt_stat_min_count": 16},
        "grouped_prompts": {**common, **grouped},
        "scaled_logit": {**common, "evasion_mode": "scaled_logit"},
        **{f"combined_ref_{label}": {**common, **combined, "reward_reference_weight": weight}
           for label, weight in (("000", 0.0), ("003", 0.03), ("010", 0.1))},
    }


def split_rows(data):
    train, validation, test = data["fitting"][:800], data["fitting"][800:], data["evaluation"]
    sets = [{row["sample_id"] for row in rows} for rows in (train, validation, test)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Attacker train, validation and test prompts overlap")
    if [len(rows) for rows in (train, validation, test)] != [800, 200, 500]:
        raise ValueError("Unexpected frozen attacker split sizes")
    return train, validation, test


def scaled_logit_values(logits, center, scale):
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid training-only logit calibration")
    values = torch.as_tensor(logits, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite detector logits")
    return torch.sigmoid((-values - center) / scale).numpy()


class ScaledLogitReward(ImageDetectorReward):
    def __init__(self, detector, calibration):
        super().__init__(detector, name="dda")
        self.calibration = calibration

    def __call__(self, image):
        return float(scaled_logit_values(self.detector.predict_logits([image]),
                                        self.calibration["center"], self.calibration["scale"])[0])


def trainer_for(manifest, name, long=False):
    root = Path(manifest["directory"])
    data = json.loads(Path(manifest["partitions"]).read_text())
    split_rows(data)
    variant = dict(manifest["variants"][name])
    mode = variant.pop("evasion_mode", "probability")
    config = dict(manifest["attacker_config"])
    config.update(variant)
    directory = root / ("long" if long else name)
    config.update(output_dir=str(directory / "attacker"),
                  reference_kl_scale=manifest["reference_scale"])
    if long:
        config.update(num_train_epochs=12800 // (config["sample_batch_size"] * config["rollout_batches"]),
                      validation_interval=800 // (config["sample_batch_size"] * config["rollout_batches"]))
    cfg = ImageEvasionConfig(**config)
    research.write_json(directory / "effective_config.json", {**asdict(cfg), "evasion_mode": mode})
    trainer = research.image_trainer(manifest, cfg, data)
    trainer.eval_prompts = [row["prompt"] for row in data["fitting"][800:]]
    if mode == "scaled_logit":
        detector = trainer.reward_fn.detector_rewards[0].detector
        calibration = json.loads((root / "logit_calibration.json").read_text())
        trainer.reward_fn.detector_rewards = [ScaledLogitReward(detector, calibration)]
    return trainer, data, directory


def evaluate(trainer, prompts, directory, stop, checkpoint_identity, *, score=True, seed=20261025):
    directory = Path(directory)
    signature = {"prompts": research.digest(prompts), "checkpoint": checkpoint_identity,
                 "sampler": "float32_trajectory_DDIM_eta1_CFG7.5", "seed": seed}
    if not score:
        signature["score"] = False
    cache = directory / "evaluation.pt"
    state = load_checkpoint(cache, signature) or {"config": signature, "scores": [], "clip": [], "images": []}
    saved_rng = rng_state()
    trainer.pipeline.unet.eval()
    detector = trainer.reward_fn.detector_rewards[0].detector
    try:
        for start in range(len(state["images"]), len(prompts), 8):
            if stop.requested:
                raise CheckpointPause("Validation image progress saved")
            batch = prompts[start:start + 8]
            embeds, negative = trainer._encode_prompts([row["prompt"] for row in batch])
            generator = torch.Generator(device=trainer.pipeline.unet.device).manual_seed(seed + start)
            with torch.no_grad():
                latents, _, _ = pipeline_sample_with_logprob(trainer.pipeline, embeds, negative,
                                                            generator=generator, num_inference_steps=50)
                images = decode_latents_to_pil(trainer.pipeline, latents)
            scores = detector.predict_batch(images).tolist() if score else []
            clips = [trainer.reward_fn._get_clip_score(image, row["prompt"]) for image, row in zip(images, batch)] if score else []
            directory.mkdir(parents=True, exist_ok=True)
            for offset, (image, row) in enumerate(zip(images, batch)):
                image_path = directory / f"{start + offset:05d}.png"
                image.save(image_path)
                state["images"].append({"sample_id": row["sample_id"], "path": str(image_path),
                                        "sha256": research.file_hash(image_path)})
            state["scores"].extend(scores)
            state["clip"].extend(clips)
            atomic_save(state, cache)
    finally:
        restore_rng(saved_rng)
    if not score:
        result = {"signature": signature, "images": state["images"]}
        research.write_json(directory / "images.json", result)
        return result
    result = {"num_images": len(prompts), "mean_evasion": float(1 - np.mean(state["scores"])),
              "evasion_rate": float(np.mean(np.asarray(state["scores"]) < 0.5)),
              "clip_mean": float(np.mean(state["clip"])), "signature": signature}
    research.write_json(directory / "metrics.json", result)
    return result


def selection_key(record):
    return record["evasion_rate"], record["mean_evasion"], record["clip_mean"]


def initialize_selection(trainer, directory, baseline):
    directory = Path(directory)
    if not (directory / "selected.json").exists():
        atomic_save({"model": model_state(trainer.pipeline.unet), "config": asdict(trainer.config)}, directory / "selected.pt")
        research.write_json(directory / "selected.json", {**baseline, "epoch": 0, "images_seen": 0, "clip_eligible": True})


def validation_callback(directory, prompts, stop, baseline_clip, minimum_clip_fraction):
    directory = Path(directory)
    def callback(trainer, epoch):
        identity = research.digest({"config": asdict(trainer.config), "epoch": epoch})
        result = evaluate(trainer, prompts, directory / "validation" / f"epoch-{epoch:05d}", stop, identity)
        result.update(epoch=epoch, images_seen=epoch * trainer.config.sample_batch_size * trainer.config.rollout_batches,
                      clip_eligible=bool(result["clip_mean"] >= minimum_clip_fraction * baseline_clip))
        if result["images_seen"] in {6400, 12800}:
            atomic_save({"model": model_state(trainer.pipeline.unet), "config": asdict(trainer.config)},
                        directory / f"budget-{result['images_seen']}.pt")
            research.write_json(directory / f"budget-{result['images_seen']}.json", result)
        selection = directory / "selected.json"
        previous = json.loads(selection.read_text()) if selection.exists() else None
        if result["clip_eligible"] and (previous is None or selection_key(result) > selection_key(previous)):
            atomic_save({"model": model_state(trainer.pipeline.unet), "config": asdict(trainer.config)}, directory / "selected.pt")
            research.write_json(selection, result)
        research.write_json(directory / "latest_validation.json", result)
        print("Attacker validation", directory.name, epoch, result, flush=True)
    return callback


def diagnostic(manifest, stop):
    root = Path(manifest["directory"])
    trainer, data, directory = trainer_for(manifest, "control")
    training, validation, _ = split_rows(data)
    original = Path(manifest["parent_directory"])
    rows = json.loads((original / "base_fitting/images.json").read_text())["rows"][:256]
    if any(row["sample_id"] not in {r["sample_id"] for r in training} for row in rows):
        raise ValueError("Logit calibration crossed the training boundary")
    detector = trainer.reward_fn.detector_rewards[0].detector
    logits = []
    for start in range(0, len(rows), 8):
        with_images = []
        for row in rows[start:start + 8]:
            if research.file_hash(row["path"]) != row["sha256"]:
                raise ValueError("Calibration image identity changed")
            with Image.open(row["path"]) as image:
                with_images.append(image.convert("RGB"))
        logits.extend(detector.predict_logits(with_images).tolist())
    values = -np.asarray(logits)
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training logits have no measurable spread")
    research.write_json(root / "logit_calibration.json", {"center": float(np.median(values)), "scale": scale,
                         "training_sample_ids": [row["sample_id"] for row in rows], "raw_ai_logits": logits})
    baseline = evaluate(trainer, validation, root / "baseline_validation", stop, "initial_SD1.5")
    embeds, negative = trainer._encode_prompts([row["prompt"] for row in training[:8]])
    trainer.pipeline.unet.eval()
    with torch.no_grad():
        _, latents, log_probs = pipeline_sample_with_logprob(trainer.pipeline, embeds, negative,
                                                            num_inference_steps=50,
                                                            generator=torch.Generator(device=trainer.pipeline.unet.device).manual_seed(42))
    errors = []
    trainer.pipeline.unet.train()
    for index, timestep in enumerate(trainer.pipeline.scheduler.timesteps):
        t = int(timestep)
        if float(trainer.pipeline.scheduler._get_variance(t, previous_ddim_timestep(trainer.pipeline.scheduler, t))) <= 0:
            continue
        _, replayed = ddim_step_with_logprob(trainer.pipeline.scheduler, trainer.pipeline.unet, latents[index], t,
                                             torch.cat([negative, embeds]), prev_sample=latents[index + 1])
        errors.append(float((replayed.detach() - log_probs[index]).abs().max()))
        if len(errors) == 1:
            advantages = torch.linspace(-1, 1, len(replayed), device=replayed.device)
            (-advantages * torch.exp(replayed - log_probs[index])).mean().backward()
            norm = float(torch.nn.utils.clip_grad_norm_(trainer.pipeline.unet.parameters(), 1.0, error_if_nonfinite=True))
            trainer.pipeline.unet.zero_grad()
    result = {"max_replay_logprob_error": max(errors), "initial_gradient_norm": norm,
              "stochastic_steps_checked": len(errors), "baseline_validation": baseline}
    research.write_json(root / "diagnostic_metrics.json", result)
    if max(errors) >= 1e-5 or not np.isfinite(norm) or norm <= 0:
        raise RuntimeError(f"Policy replay/gradient diagnostic failed: {result}")
    del latents, log_probs, trainer
    gc.collect()
    torch.cuda.empty_cache()


def pilot(manifest, name, stop):
    trainer, data, directory = trainer_for(manifest, name)
    _, validation, _ = split_rows(data)
    baseline = json.loads((Path(manifest["directory"]) / "baseline_validation/metrics.json").read_text())
    initialize_selection(trainer, directory, baseline)
    trainer.validation_callback = validation_callback(directory, validation, stop, baseline["clip_mean"], manifest["minimum_clip_fraction"])
    trainer.train()
    research.write_json(directory / "training_history.json", trainer.training_history)
    if not (directory / "selected.json").exists():
        research.write_json(directory / "ineligible.json", {"reason": "No checkpoint met the validation CLIP floor"})


def choose_pilot(manifest):
    root = Path(manifest["directory"])
    candidates = {name: json.loads((root / name / "selected.json").read_text()) for name in manifest["variants"]
                  if (root / name / "selected.json").exists()}
    if not candidates:
        raise RuntimeError("No pilot met the predeclared CLIP floor")
    winner = max(sorted(candidates), key=lambda name: (*selection_key(candidates[name]), name == "control"))
    research.write_json(root / "pilot_selection.json", {"winner": winner, "candidates": candidates,
                                                       "rule": "CLIP floor, then validation evasion rate, mean evasion, CLIP"})
    return winner


def long_run(manifest, stop):
    root = Path(manifest["directory"])
    selection = root / "pilot_selection.json"
    winner = json.loads(selection.read_text())["winner"] if selection.exists() else choose_pilot(manifest)
    trainer, data, directory = trainer_for(manifest, winner, long=True)
    _, validation, _ = split_rows(data)
    baseline = json.loads((root / "baseline_validation/metrics.json").read_text())
    initialize_selection(trainer, directory, baseline)
    trainer.validation_callback = validation_callback(directory, validation, stop, baseline["clip_mean"], manifest["minimum_clip_fraction"])
    trainer.train()
    research.write_json(directory / "training_history.json", trainer.training_history)
    if not (directory / "selected.pt").exists():
        raise RuntimeError("Long run produced no CLIP-eligible checkpoint")


def final_test(manifest, stop):
    root = Path(manifest["directory"])
    winner = json.loads((root / "pilot_selection.json").read_text())["winner"]
    trainer, data, directory = trainer_for(manifest, winner, long=True)
    _, _, test = split_rows(data)
    baseline = evaluate(trainer, test, root / "test/initial", stop, "initial_SD1.5")
    checkpoint = load_checkpoint(directory / "selected.pt", trainer.config)
    restore_model(trainer.pipeline.unet, checkpoint["model"])
    attacked = evaluate(trainer, test, root / "test/selected", stop, research.file_hash(directory / "selected.pt"))
    detector = trainer.reward_fn.detector_rewards[0].detector
    human = research.score_images(detector, data["real_evaluation"], root / "test/human.pt", stop)
    for name, result in (("initial", baseline), ("selected", attacked)):
        scores = load_checkpoint(root / f"test/{name}/evaluation.pt")["scores"]
        result["classification"] = research.metrics([{ "label": 1}] * len(scores) + data["real_evaluation"],
                                                     np.concatenate([scores, human]))
    research.write_json(root / "test_results.json", {"initial": baseline, "selected": attacked,
                                                     "pilot": winner, "selected_on_validation_only": True})


def evaluate_checkpoint(manifest, stop):
    parent = json.loads(Path(manifest["checkpoint_parent_manifest"]).read_text())
    parent_root = Path(parent["directory"])
    for name in ("long", "test"):
        receipt = json.loads((parent_root / f"{name}.json").read_text())
        if receipt.get("status") != "completed" or receipt.get("job_id") != parent["jobs"][name]["job_id"]:
            raise ValueError("Original training and test jobs must have completed")
    if research.file_hash(parent["partitions"]) != manifest["partitions_sha256"]:
        raise ValueError("Held-out prompt partitions changed")
    if research.file_hash(parent_root / "warmup/completed.pt") != manifest["warmup_sha256"]:
        raise ValueError("Warmed defender changed")
    checkpoint_path = Path(manifest["evaluation_checkpoint"])
    if research.file_hash(checkpoint_path) != manifest["evaluation_checkpoint_sha256"]:
        raise ValueError("Requested checkpoint changed")
    baseline_path = parent_root / "test_results.json"
    if research.file_hash(baseline_path) != manifest["baseline_test_sha256"]:
        raise ValueError("Saved baseline test results changed")
    data = json.loads(Path(parent["partitions"]).read_text())
    _, _, prompts = split_rows(data)
    baseline = json.loads(baseline_path.read_text())["initial"]
    expected = {"prompts": research.digest(prompts), "checkpoint": "initial_SD1.5",
                "sampler": "float32_trajectory_DDIM_eta1_CFG7.5", "seed": 20261025}
    if baseline["signature"] != expected or baseline["num_images"] != 500:
        raise ValueError("Baseline test uses different prompts or sampling")
    config = ImageEvasionConfig(**manifest["attacker_config"])
    checkpoint = load_checkpoint(checkpoint_path, config)
    root = Path(manifest["directory"])
    trainer = research.image_trainer(parent, replace(config, output_dir=str(root / "generator")), data)
    restore_model(trainer.pipeline.unet, checkpoint["model"])
    attacked = evaluate(trainer, prompts, root / "evaluated", stop, manifest["evaluation_checkpoint_sha256"])
    cached = load_checkpoint(root / "evaluated/evaluation.pt")
    human = load_checkpoint(parent_root / "test/human.pt")
    if (human is None or human["config"]["checkpoint"] != manifest["warmup_sha256"]
            or human["config"]["rows"] != research.digest(data["real_evaluation"]) or len(human["scores"]) != 500):
        raise ValueError("Human scores do not match the warmed defender and held-out split")
    labels = [{"label": 1}] * len(prompts) + data["real_evaluation"]
    scores = np.concatenate([cached["scores"], human["scores"]])
    attacked["classification"] = research.metrics(labels, scores)
    threshold = json.loads((parent_root / "warmup/threshold.json").read_text())["threshold"]
    attacked["warmup_validation_threshold"] = research.metrics(labels, scores, threshold)
    attacked["clip_retention_vs_initial"] = attacked["clip_mean"] / baseline["clip_mean"]
    research.write_json(root / "results.json", {"initial": baseline, "evaluated": attacked,
                                               "checkpoint_sha256": manifest["evaluation_checkpoint_sha256"],
                                               "post_hoc_evaluation": True, "training_performed": False})


def defender_inputs(manifest, data, stop):
    from ..checkpointing import restore_model
    root = Path(manifest["directory"])
    parent = json.loads(Path(manifest["attacker_parent_manifest"]).read_text())
    parent_root = Path(parent["directory"])
    split_rows(data)
    if research.file_hash(parent_root / "warmup/completed.pt") != parent["warmup_sha256"]:
        raise ValueError("Parent warmed defender changed")
    if research.file_hash(parent["partitions"]) != parent["partitions_sha256"]:
        raise ValueError("Parent partitions changed")
    if {row["sha256"] for row in data["real_fitting"]} & {row["sha256"] for row in data["real_evaluation"]}:
        raise ValueError("Human fitting and held-out images overlap")
    config = json.loads((parent_root / "long/effective_config.json").read_text())
    config.pop("evasion_mode")
    config = ImageEvasionConfig(**config)
    selected = parent_root / "long/selected.pt"
    identity = research.file_hash(selected)
    checkpoint = load_checkpoint(selected, config)
    result = json.loads((parent_root / "test_results.json").read_text())
    if result["selected"]["signature"]["checkpoint"] != identity:
        raise ValueError("Defender must use the same attacker as the completed test job")
    signature = {"attacker_sha256": identity, "warmup_sha256": parent["warmup_sha256"],
                 "partitions": research.digest(data)}
    previous = root / "defender_inputs.json"
    if previous.exists() and json.loads(previous.read_text()) != signature:
        raise ValueError("Defender inputs changed on resume")
    research.write_json(previous, signature)
    def rows(prompts, images):
        if len(prompts) != len(images) or [p["sample_id"] for p in prompts] != [p["sample_id"] for p in images]:
            raise ValueError("Saved images do not match the ordered prompt split")
        output = [{**prompt, **image, "label": 1} for prompt, image in zip(prompts, images)]
        for row in output:
            if research.file_hash(row["path"]) != row["sha256"]:
                raise ValueError("Saved evaluation image identity changed")
        return output
    evaluation = {}
    for name, model in (("initial", "initial_SD1.5"), ("selected", identity)):
        expected = {"prompts": research.digest(data["evaluation"]), "checkpoint": model,
                    "sampler": "float32_trajectory_DDIM_eta1_CFG7.5", "seed": 20261025}
        cached = load_checkpoint(parent_root / f"test/{name}/evaluation.pt", expected)
        if cached is None or len(cached["scores"]) != len(data["evaluation"]):
            raise ValueError("Parent evaluation is incomplete")
        evaluation[name] = rows(data["evaluation"], cached["images"])
    trainer = research.image_trainer(parent, replace(config, output_dir=str(root / "generator")), data)
    restore_model(trainer.pipeline.unet, checkpoint["model"])
    generated = evaluate(trainer, data["fitting"], root / "attacked_fitting", stop, identity,
                         score=False, seed=20261026)
    fitting = rows(data["fitting"], generated["images"])
    del trainer, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return fitting, evaluation["selected"], evaluation["initial"]
