"""Released DistSeal generators and forgetting without original generator weights."""

from __future__ import annotations

import gc
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ..rl_evasion.checkpointing import (CheckpointClock, CheckpointPause, atomic_save,
                                      model_state, restore_model, rng_state, restore_rng)
from ..rl_evasion.config import seed_everything
from .protocol import detection_metrics, digest, operating_point, write_json


def cards(manifest, family, watermarked=True):
    root = Path(manifest["resources"]["cards"])
    generator = ("rar_xl" if family == "rar" else "uvit-h") + ("_distseal" if watermarked else "_base")
    decoder = "maskgit_base" if family == "rar" else "dc-ae-f64c128_base"
    detector = "detector_rar" if family == "rar" else "detector_dc-ae"
    return tuple(str(root / f"{name}.yaml") for name in (generator, decoder, detector))


def labels_for(manifest, count, calibration=False):
    labels = np.random.default_rng(manifest["seed"] + (1000 if calibration else 2000)).integers(1, 1000, count)
    labels[:min(manifest["image_evaluation"]["class_zero_n"], count//4)] = 0
    return labels.tolist()


def bit_scores(detector, message, images):
    predictions = detector.detect_watermark(images)
    if predictions.ndim != 2 or predictions.shape[-1] != message.numel():
        raise ValueError(f"Unexpected watermark output: {predictions.shape}, key {message.shape}")
    return (predictions.bool() == message.bool().reshape(1, -1)).float().mean(-1).cpu().tolist()


@torch.no_grad()
def sample(manifest, family, generator, decoder, detector, message, clip, processor,
           destination, count, stop, calibration=False):
    from torchvision.models import ResNet50_Weights
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "samples.json"
    labels = labels_for(manifest, count, calibration)
    signature = digest({"labels": labels, "protocol": manifest["protocol_sha256"],
                        "calibration": calibration, "family": family, "destination": str(destination)})
    cache = json.loads(output.read_text()) if output.exists() else {"signature": signature, "rows": []}
    if cache["signature"] != signature:
        raise ValueError("Image sample cache does not match protocol")
    categories = ResNet50_Weights.DEFAULT.meta["categories"]
    batch_size = manifest["image_evaluation"]["batch"]
    for start in range(len(cache["rows"]), count, batch_size):
        batch_labels = torch.tensor(labels[start:start+batch_size], device="cuda", dtype=torch.long)
        seed_everything(manifest["seed"] + (10000 if calibration else 20000) + start)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if family == "rar":
                tokens = generator.generate(batch_labels, guidance_scale=6.9, guidance_scale_pow=1.5,
                                            randomize_temperature=1.02)
                images = decoder.decode(tokens).float().clamp(0, 1).mul(255).to(torch.uint8)
            else:
                latents = generator.generate(batch_labels.int(), torch.full_like(batch_labels, 1000).int(), scale=1.)
                images = decoder.decode(latents / .2889).float().mul(127.5).add(128).clamp(0, 255).to(torch.uint8)
        pil = [Image.fromarray(image.permute(1, 2, 0).cpu().numpy()) for image in images]
        from torchvision.transforms import functional as TF
        tensor = images.float()/255
        bit_accuracy = bit_scores(detector, message, tensor)
        jpeg = []
        for image in pil:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=50)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                jpeg.append(TF.pil_to_tensor(compressed.convert("RGB")))
        attack_scores = {
            "jpeg50": bit_scores(detector, message, torch.stack(jpeg).cuda().float()/255),
            "center_crop_area50": bit_scores(detector, message, TF.center_crop(tensor, int(images.shape[-1]*.5**.5))),
        }
        texts = ["a photo of a " + categories[y] for y in batch_labels.cpu().tolist()]
        inputs = processor(images=pil, text=texts, return_tensors="pt", padding=True).to("cuda")
        outputs = clip(**inputs)
        similarities = (outputs.image_embeds * outputs.text_embeds).sum(-1).cpu().tolist()
        for index, (image, bitacc, alignment) in enumerate(zip(pil, bit_accuracy, similarities), start):
            image.save(destination / f"{index:05}.png")
            cache["rows"].append({"index": index, "label": labels[index], "bit_accuracy": bitacc,
                                  "clip_alignment": alignment,
                                  "attacks": {k: v[index-start] for k,v in attack_scores.items()}})
        write_json(output, cache)
        if stop.requested:
            raise CheckpointPause("Image sample batch saved")
    return cache["rows"]


def summarize(rows, null, threshold, attack_thresholds=None):
    def metrics(selected, negatives):
        scores = [r["bit_accuracy"] for r in selected]
        result = detection_metrics(scores, [r["bit_accuracy"] for r in negatives], threshold)
        result.update(mean_bit_accuracy=float(np.mean(scores)),
                      clip_alignment=float(np.mean([r["clip_alignment"] for r in selected])))
        result["attacks"] = {
            name: {**detection_metrics([r["attacks"][name] for r in selected],
                                      [r["attacks"][name] for r in negatives], value),
                   "mean_bit_accuracy": float(np.mean([r["attacks"][name] for r in selected]))}
            for name, value in (attack_thresholds or {}).items()}
        return result
    return {"all": metrics(rows, null),
            "class_zero": metrics([r for r in rows if r["label"] == 0], [r for r in null if r["label"] == 0]),
            "other_classes": metrics([r for r in rows if r["label"] != 0], [r for r in null if r["label"] != 0])}


def fit_clean(manifest, family, generator, decoder, stop, callback):
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
    config = manifest["image_training"]
    if family == "rar":
        from deps.rar.modeling.modules.losses import ARLoss
        ar_loss = ARLoss(generator.config)
    output = Path(manifest["directory"]) / f"{family}_clean"
    output.mkdir(exist_ok=True)
    data = json.loads(Path(manifest["data"]).read_text())["images"]
    size = 256 if family == "rar" else 512
    transform = Compose([Resize(size), CenterCrop(size), ToTensor()])
    latent_path = output / "fitting_latents.pt"
    if latent_path.exists():
        encoded = torch.load(latent_path, weights_only=True)
    else:
        from distseal.modules.lora_helper import apply_lora_to_linear, mark_only_lora_as_trainable
        values = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in data["train"]:
                path = row["path"] if isinstance(row, dict) else row
                with Image.open(path) as image:
                    tensor = transform(image.convert("RGB")).unsqueeze(0).cuda()
                latent = decoder.encode(tensor if family == "rar" else tensor*2-1)
                values.append(latent.cpu())
        encoded = torch.cat(values)
        atomic_save(encoded, latent_path)
    # The decoder is a component of the released pipeline; no original generator,
    # watermark key or detector is used by this optimization.
    if config.get("mode") == "full":
        generator.requires_grad_(True)
        parameters = list(generator.parameters())
    elif config.get("generic", False):
        from .backbones import inject_lora
        inject_lora(generator, config["lora_rank"])
        parameters = [p for p in generator.parameters() if p.requires_grad]
    else:
        apply_lora_to_linear(generator, r=config["lora_rank"], lora_alpha=32, target_linear_modules=("",))
        parameters = list(mark_only_lora_as_trainable(generator))
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=0., foreach=False)
    step, history = 0, []
    restart = output / "restart.pt"
    if restart.exists():
        state = torch.load(restart, map_location="cpu", weights_only=False)
        if state["protocol"] != manifest["protocol_sha256"]:
            raise ValueError("Image forgetting restart does not match protocol")
        restore_model(generator, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        step, history = state["step"], state["history"]
        del state
    clock = CheckpointClock(600)
    def save():
        atomic_save({"protocol": manifest["protocol_sha256"], "step": step,
                     "model": model_state(generator), "optimizer": optimizer.state_dict(),
                     "rng": rng_state(), "history": history}, restart)
        write_json(output / "progress.json", {"step": step, "history": history,
                                              "original_generator_weights_access": False,
                                              "watermark_detector_used_in_loss": False})
        clock.saved()
    def evaluate_saved_step():
        training_rng = rng_state()
        generator.eval()
        try:
            callback(f"clean_step_{step}")
        finally:
            restore_rng(training_rng)
    export_steps = config.get("export_steps", [500, 2500])
    if step in export_steps:
        evaluate_saved_step()
    while step < config["steps"]:
        generator.train()
        if family == "rar":
            generator.disable_kv_cache()
        indices = np.random.default_rng(manifest["seed"]+step).integers(len(encoded), size=config["batch"])
        latent = encoded[indices].cuda()
        labels = torch.tensor([data["train"][i]["label"] if isinstance(data["train"][i], dict) else 0
                               for i in indices], dtype=torch.long, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if family == "rar":
                logits, target = generator(latent, generator.preprocess_condition(labels), return_labels=True)
                loss, _ = ar_loss(logits.float(), target)
            else:
                loss, _ = generator(latent * .2889, labels.int())
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite clean image training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite clean image gradient")
        optimizer.step()
        step += 1
        if step % 50 == 0:
            history.append({"step": step, "loss": float(loss.detach())})
            print(family, "forgetting", history[-1], flush=True)
        if step % 500 == 0 or clock.due() or stop.requested:
            save()
        if stop.requested:
            raise CheckpointPause(f"Image clean training saved at step {step}")
        if step in export_steps:
            save()
            evaluate_saved_step()
    save()


def evaluate(manifest, stage, stop):
    from distseal.loader import load_generator, load_decoder, load_detector
    from transformers import CLIPModel, CLIPProcessor
    family = stage.split("_")[0]
    root = Path(manifest["directory"])
    output = root / stage
    output.mkdir(exist_ok=True)
    generator_card, decoder_card, detector_card = cards(manifest, family)
    generator = load_generator(generator_card, device="cuda")
    decoder = load_decoder(decoder_card, device="cuda").eval().requires_grad_(False)
    detector, message = load_detector(detector_card, device="cuda")
    detector = detector.eval().requires_grad_(False)
    clip = CLIPModel.from_pretrained(manifest["clip"]["path"], local_files_only=True).cuda().eval()
    processor = CLIPProcessor.from_pretrained(manifest["clip"]["path"], local_files_only=True)
    config = manifest["image_evaluation"]
    def generate(name, count=None, calibration=False):
        return sample(manifest, family, generator, decoder, detector, message, clip, processor,
                      output / name, config["n"] if count is None else count, stop, calibration)
    if stage == family:
        watermarked = generate("watermarked")
        del generator
        gc.collect(); torch.cuda.empty_cache()
        generator = load_generator(cards(manifest, family, watermarked=False)[0], device="cuda")
        calibration = generate("base_calibration", config["calibration_n"], calibration=True)
        null = generate("base")
        threshold = operating_point([r["bit_accuracy"] for r in calibration])
        attack_thresholds = {name: operating_point([r["attacks"][name] for r in calibration])
                             for name in ("jpeg50", "center_crop_area50")}
        results = {"threshold": threshold, "nominal_fpr": .01, "threshold_source": "independent base-model seeds",
                   "attack_thresholds": attack_thresholds, "watermark_location": "generator", "conditions": {
                   "watermarked": summarize(watermarked, null, threshold, attack_thresholds),
                   "base": summarize(null, null, threshold, attack_thresholds)}}
    else:
        baseline = json.loads((root / family / "results.json").read_text())
        threshold = baseline["threshold"]
        attack_thresholds = baseline["attack_thresholds"]
        null = json.loads((root / family / "base/samples.json").read_text())["rows"]
        results = {"threshold": threshold, "conditions": {},
                   "original_generator_weights_access": stage.endswith("_base"),
                   "watermark_detector_used_in_loss": False}
        if (output / "results.json").exists():
            results["conditions"].update(json.loads((output / "results.json").read_text())["conditions"])
        if stage.endswith("_base"):
            base = load_generator(cards(manifest, family, watermarked=False)[0], device="cpu")
            original = {k: p.detach().cpu().clone() for k, p in generator.named_parameters()}
            clean = dict(base.named_parameters())
            for weight in (.75, .5, .25, 0.):
                with torch.no_grad():
                    for key, p in generator.named_parameters():
                        p.copy_(weight*original[key].to(p.device) + (1-weight)*clean[key].to(p.device))
                name = f"watermarked_weight_{weight:g}"
                rows = generate(name)
                results["conditions"][name] = summarize(rows, null, threshold, attack_thresholds)
                if weight == 0. and not np.allclose([r["bit_accuracy"] for r in rows], [r["bit_accuracy"] for r in null]):
                    raise ValueError("Full base replacement does not reproduce the saved base scores")
                write_json(output / "results.json", results)
        else:
            def callback(name):
                rows = generate(name)
                results["conditions"][name] = summarize(rows, null, threshold, attack_thresholds)
                write_json(output / "results.json", results)
            fit_clean(manifest, family, generator, decoder, stop, callback)
    write_json(output / "results.json", results)
