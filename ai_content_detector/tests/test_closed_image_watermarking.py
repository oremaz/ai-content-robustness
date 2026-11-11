"""Closed-source attack contracts, artifact integrity and resumable pilot coverage."""

import copy
import json
from pathlib import Path

from PIL import Image
import pytest

from ai_content_detector.watermarking import closed_image as module


def task_for(tmp_path):
    images = []
    for i in range(4):
        path = tmp_path/f"{i}.png"
        Image.new("RGB", (256, 256), color=(i*30, 20, 80)).save(path)
        images.append({"id": str(i), "path": str(path), "sha256": module.file_hash(path), "size": [256, 256]})
    task = {"directory": str(tmp_path/"outputs"), "source": "source", "upstream": "public_nfpa",
            "public_model": {"path": "public_sd", "name": "independent_model", "revision": "pinned"}, "images": images,
            "recipe": {"steps": 10, "deltas": [0, 20, 40], "primary_delta": 40, "seed": 42,
                       "input_size": 512, "prompt": "", "guidance_scale": 7.5, "resize": "Lanczos",
                       "resize_control": True, "pilot_n": 2, "sanity_positions": [0, 1]}}
    task["signature"] = module.digest(task)
    return task


def resign(task):
    task["signature"] = module.digest({k: v for k, v in task.items() if k != "signature"})


def test_private_fields_rejected_at_task_recipe_and_image_levels(tmp_path):
    original = task_for(tmp_path)
    module.validate_task(original)
    for location in (None, "recipe", "image"):
        task = copy.deepcopy(original)
        target = task if location is None else task["recipe"] if location == "recipe" else task["images"][0]
        target["secret_key"] = 42
        resign(task)
        with pytest.raises(ValueError, match="private fields"):
            module.validate_task(task)
    original["recipe"]["seed"] = 9
    with pytest.raises(ValueError, match="changed"):
        module.validate_task(original)


def test_resize_geometry_and_input_integrity(tmp_path):
    task = task_for(tmp_path)
    row = task["images"][0]
    image = module.read_rgb(row)
    assert module.resize_for_attack(image).size == (512, 512)
    assert module.restore_size(module.resize_for_attack(image), image.size).size == (256, 256)
    Image.new("RGB", (256, 256), "white").save(row["path"])
    with pytest.raises(ValueError, match="changed"):
        module.read_rgb(row)


def test_pilot_is_sealed_and_full_run_reuses_its_images(tmp_path, monkeypatch):
    task = task_for(tmp_path)
    monkeypatch.setattr(module, "load_pipeline", lambda task: object())
    calls = []
    def generate(pipeline, image, steps, delta, seed, **kwargs):
        calls.append((delta, seed))
        return image.copy()
    monkeypatch.setattr(module, "nfpa_image", generate)
    class Stop:
        def check(self):
            pass
    module.attack(task, Stop(), pilot=True)
    first = module.verify_outputs(task, pilot=True)
    assert len(calls) == 6
    module.attack(task, Stop())
    assert len(calls) == 12
    assert module.verify_outputs(task, pilot=True) == first
    assert len(module.verify_outputs(task)) == 4
    path = Path(task["directory"])/"nfpa_40"/"0.png"
    Image.new("RGB", (256, 256), "white").save(path)
    with pytest.raises(ValueError, match="changed"):
        module.verify_outputs(task)


def test_attacker_child_receives_task_without_owner_manifest(tmp_path, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/"scripts"))
    import closed_image_watermark_experiments as driver
    config = {"source": "/frozen", "python": "python", "cases": [{"name": "model01", "task": "/opaque/task.json"}]}
    monkeypatch.setattr(driver, "verify", lambda config: None)
    calls = []
    class Process:
        def __init__(self, args):
            calls.append(args)
        def wait(self):
            return 0
    monkeypatch.setattr(driver.subprocess, "Popen", Process)
    driver.run(config, "/private/owner.json", "model01")
    assert len(calls) == 4
    assert calls[0][-2:] == ["pilot_attack", "/opaque/task.json"]
    assert calls[2][-2:] == ["attack", "/opaque/task.json"]
    assert all("/private/owner.json" not in call for call in (calls[0], calls[2]))
    assert calls[1][-1] == "pilot_owner:model01"
    assert calls[3][-1] == "owner:model01"
