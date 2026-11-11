"""Image-only NFPA workers, separated from watermark owner evaluation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import time

from PIL import Image


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def validate_task(task):
    allowed = {"directory", "source", "upstream", "public_model", "images", "recipe", "signature"}
    if set(task) != allowed:
        raise ValueError("Image-only attack task contains unexpected or private fields")
    if digest({k: v for k, v in task.items() if k != "signature"}) != task["signature"]:
        raise ValueError("Image-only attack task changed")
    ids = [row["id"] for row in task["images"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs")
    allowed_recipe = {"steps", "deltas", "primary_delta", "seed", "input_size", "prompt", "guidance_scale", "resize", "resize_control", "pilot_n", "sanity_positions"}
    if set(task["recipe"]) != allowed_recipe or set(task["public_model"]) - {"path", "name", "revision", "original_name", "source_note", "files_sha256"}:
        raise ValueError("Attack recipe or public model contains unexpected or private fields")
    for row in task["images"]:
        if set(row) != {"id", "path", "sha256", "size"}:
            raise ValueError("Image-only task rows contain unexpected or private fields")


def read_rgb(row):
    if file_hash(row["path"]) != row["sha256"]:
        raise ValueError(f"Input image changed: {row['id']}")
    with Image.open(row["path"]) as source:
        image = source.convert("RGB")
    if list(image.size) != row["size"]:
        raise ValueError("Image dimensions differ from the frozen task")
    return image


def resize_for_attack(image):
    return image.resize((512, 512), Image.Resampling.LANCZOS)


def restore_size(image, size):
    return image.convert("RGB").resize(tuple(size), Image.Resampling.LANCZOS)


def load_pipeline(task):
    import torch
    from diffusers import DDIMScheduler
    for name, expected in task["public_model"].get("files_sha256", {}).items():
        if file_hash(Path(task["public_model"]["path"])/name) != expected:
            raise ValueError("Independent public regeneration weights changed")
    spec = importlib.util.spec_from_file_location("nfpa_released", Path(task["upstream"])/"utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pipeline = module.MyStableDiffusionPipeline.from_pretrained(
        task["public_model"]["path"], torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True, local_files_only=True, safety_checker=None,
        feature_extractor=None, requires_safety_checker=False).to("cuda")
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)
    for model in (pipeline.vae, pipeline.text_encoder, pipeline.unet):
        model.eval().requires_grad_(False)
    return pipeline


def nfpa_image(pipeline, image, steps, delta, seed, guidance_scale=7.5):
    import torch
    from diffusers import DDIMScheduler, DDIMInverseScheduler
    from torchvision.transforms.functional import to_tensor
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    original_size = image.size
    tensor = to_tensor(resize_for_attack(image)).unsqueeze(0).to("cuda", torch.float16)*2-1
    with torch.inference_mode():
        latent = pipeline.vae.encode(tensor).latent_dist.mean*pipeline.vae.config.scaling_factor
        pipeline.scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
        inverted = pipeline(prompt="", negative_prompt="", num_inference_steps=steps,
                            guidance_scale=guidance_scale, latents=latent, output_type="latent", width=512, height=512).images
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        images = pipeline(prompt="", num_images_per_prompt=2, latents=inverted,
                          guidance_scale=guidance_scale, xyz=[delta, delta, 0], num_inference_steps=steps,
                          warped_latents_timestep=torch.tensor([0], dtype=torch.long, device="cuda")).images
    if len(images) != 2:
        raise ValueError("Released NFPA pipeline must produce reference and next frames")
    return restore_size(images[1], original_size)


def attack(task, stop, sanity=False, pilot=False):
    validate_task(task)
    output = Path(task["directory"])
    output.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline(task)
    count = task["recipe"]["pilot_n"] if pilot else len(task["images"])
    rows = [task["images"][i] for i in task["recipe"]["sanity_positions"]] if sanity else task["images"][:count]
    positions = {row["id"]: i for i, row in enumerate(task["images"])}
    arms = [(f"sanity_{task['recipe']['primary_delta']}", task["recipe"]["primary_delta"])] if sanity else (
        [("resize_only", None)]+[(f"nfpa_{d}", d) for d in task["recipe"]["deltas"]])
    for arm, delta in arms:
        folder = output/arm
        folder.mkdir(exist_ok=True)
        path = folder/"images.json"
        stored = json.loads(path.read_text()) if path.exists() else {"signature": task["signature"], "rows": []}
        completed = stored["rows"]
        if stored["signature"] != task["signature"] or [x["id"] for x in completed] != [x["id"] for x in rows[:len(completed)]]:
            raise ValueError("NFPA image restart differs from its frozen inputs")
        for row in completed:
            if file_hash(folder/row["file"]) != row["sha256"]:
                raise ValueError("A completed NFPA image changed")
        for position, row in enumerate(rows[len(completed):], len(completed)):
            image = read_rgb(row)
            started = time.monotonic()
            result = (restore_size(resize_for_attack(image), image.size) if delta is None else
                      nfpa_image(pipeline, image, task["recipe"]["steps"], delta, task["recipe"]["seed"]+positions[row["id"]],
                                 guidance_scale=task["recipe"]["guidance_scale"]))
            filename = row["id"]+".png"
            temporary = folder/(filename+".tmp")
            result.save(temporary, format="PNG")
            temporary.replace(folder/filename)
            completed.append({"id": row["id"], "file": filename, "sha256": file_hash(folder/filename),
                              "seconds": time.monotonic()-started, "size": list(result.size)})
            save(path, stored)
            if len(completed) % 32 == 0:
                print(folder.name, len(completed), "/", len(rows), flush=True)
            stop.check()
    if not sanity:
        suffix = "_pilot" if pilot else ""
        sealed = {}
        for arm, _ in arms:
            index = output/arm/"images.json"
            if pilot:
                snapshot = output/arm/"images_pilot.json"
                save(snapshot, json.loads(index.read_text()))
                index = snapshot
            sealed[arm] = file_hash(index)
        save(output/("attacks_complete"+suffix+".json"), sealed)


def verify_outputs(task, pilot=False):
    validate_task(task)
    output = Path(task["directory"])
    suffix = "_pilot" if pilot else ""
    sealed = json.loads((output/("attacks_complete"+suffix+".json")).read_text())
    expected = {"resize_only"}|{f"nfpa_{d}" for d in task["recipe"]["deltas"]}
    if set(sealed) != expected:
        raise ValueError("NFPA strength sweep is incomplete")
    for arm, expected_hash in sealed.items():
        path = output/arm/("images"+suffix+".json")
        if file_hash(path) != expected_hash:
            raise ValueError("Frozen NFPA artifact index changed")
        records = json.loads(path.read_text())
        expected_rows = task["images"][:task["recipe"]["pilot_n"]] if pilot else task["images"]
        if records["signature"] != task["signature"] or [x["id"] for x in records["rows"]] != [x["id"] for x in expected_rows]:
            raise ValueError("NFPA artifacts do not cover the complete frozen input set")
        for row in records["rows"]:
            if file_hash(output/arm/row["file"]) != row["sha256"]:
                raise ValueError("Frozen NFPA image changed")
    return sealed
