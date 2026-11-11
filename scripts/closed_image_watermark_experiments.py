"""Closed-source image watermark evaluation using released NeurIPS 2025 NFPA."""

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from ai_content_detector.watermarking.closed_image import (
    attack, digest, file_hash, read_rgb, save, validate_task, verify_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT/"results/research_workflow/watermarking/resources"
ORIGINAL = ROOT/"results/research_workflow/watermarking/20261026T001727478096Z"
CHILD = None


def read(path):
    return json.loads(Path(path).read_text())


class Pause(Exception):
    pass


class Stop:
    def __init__(self):
        self.signum = None

    def request(self, signum, frame):
        self.signum = signum

    def check(self):
        if self.signum is not None:
            raise Pause("Progress saved")


def prepare():
    from PIL import Image
    owner = read(ORIGINAL/"manifest.json")
    public = read(RESOURCES/"nfpa_model.json")
    public["files_sha256"] = {str(p.relative_to(public["path"])): file_hash(p) for p in Path(public["path"]).rglob("*.safetensors")}
    upstream = RESOURCES/"nfpa"
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != "7f449f305506f95876fbfde0690395bea720269c":
        raise ValueError("NFPA source differs from the reviewed release")
    directory = ROOT/"results/research_workflow/closed_image"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    source = directory/"source"
    shutil.copytree(ROOT/"ai_content_detector", source/"ai_content_detector", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (source/"scripts").mkdir()
    for name in ("closed_image_watermark_experiments.py", "run_closed_image_watermarks.slurm"):
        shutil.copyfile(ROOT/"scripts"/name, source/"scripts"/name)
    shutil.copytree(upstream, directory/"upstream", ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    cases = []
    inputs = {}
    for number, family in enumerate(("rar", "dcae"), 1):
        name = f"model{number:02d}"
        case_dir = directory/name
        (case_dir/"inputs").mkdir(parents=True)
        images, mapping, cohorts = [], {}, {}
        for cohort, folder, count in (("calibration", "base_calibration", 256), ("clean", "base", 512), ("marked", "watermarked", 512)):
            records = read(ORIGINAL/family/folder/"samples.json")["rows"]
            if len(records) != count:
                raise ValueError(f"Incomplete preserved image cohort: {family}/{folder}")
            cohorts[cohort] = []
            for record in records:
                index = record["index"]
                original = ORIGINAL/family/folder/f"{index:05}.png"
                identity = f"image{len(images):05}"
                destination = case_dir/"inputs"/(identity+".png")
                shutil.copyfile(original, destination)
                checksum = file_hash(original)
                if file_hash(destination) != checksum:
                    raise ValueError("Copied image differs from its original")
                with Image.open(destination) as image:
                    size = list(image.size)
                row = {"id": identity, "path": str(destination), "sha256": checksum, "size": size}
                images.append(row)
                cohorts[cohort].append(row)
                mapping[identity] = {"cohort": cohort, "label": record["label"], "original_score": record["bit_accuracy"]}
        pilot_indices = {i*512//32 for i in range(32)}
        images = cohorts["calibration"]+[cohorts[c][i] for c in ("clean", "marked") for i in sorted(pilot_indices)]+[
            row for c in ("clean", "marked") for i, row in enumerate(cohorts[c]) if i not in pilot_indices]
        task = {"directory": str(case_dir/"attacks"), "source": str(source), "upstream": str(directory/"upstream"),
                "public_model": public, "images": images,
                "recipe": {"steps": 10, "deltas": [0, 20, 40], "primary_delta": 40, "seed": 20261104,
                           "input_size": 512, "prompt": "", "guidance_scale": 7.5,
                           "resize": "Lanczos to 512, then restore original resolution", "resize_control": True,
                           "pilot_n": 320, "sanity_positions": [256, 288]}}
        task["signature"] = digest(task)
        validate_task(task)
        save(case_dir/"task.json", task)
        results = read(ORIGINAL/family/"results.json")
        card = Path(owner["resources"]["cards"])/("detector_rar.yaml" if family == "rar" else "detector_dc-ae.yaml")
        cases.append({"name": name, "family": family, "task": str(case_dir/"task.json"), "mapping": mapping,
                      "detector_card": str(card), "original_threshold": results["threshold"]})
        inputs[str(case_dir/"task.json")] = file_hash(case_dir/"task.json")
    config = {"directory": str(directory), "source": str(source), "cases": cases,
              "python": str(ROOT/".venv/bin/python"), "nfpa_revision": revision,
              "paper": "https://proceedings.neurips.cc/paper_files/paper/2025/hash/b48405307576689bea14d221ccba73ae-Abstract-Conference.html",
              "paper_sha256": file_hash(RESOURCES/"closed_image_papers/nfpa_neurips2025.pdf"),
              "owner_distseal_source": owner["resources"]["repositories"]["distseal"]["path"],
              "owner_source_hashes": {p: h for p, h in owner["resource_code_sha256"].items() if Path(p).is_relative_to(owner["resources"]["repositories"]["distseal"]["path"])},
              "owner_weight_hashes": {v["path"]: v["sha256"] for k, v in owner["resources"]["files"].items() if k.startswith("detector_")},
              "clip": owner["clip"], "metric_cache": str(RESOURCES/"closed_image_metrics"),
              "access": "Attack workers receive only opaque PNGs and independent public model, no target weights/decoder/detector/keys/payload/labels/prompts.",
              "selection": "All three strengths fixed before owner scoring. No detector feedback selects images, motion or strength.",
              "input_hashes": inputs,
              "source_hashes": {str(p.relative_to(directory)): file_hash(p) for base in (source, directory/"upstream") for p in base.rglob("*") if p.is_file()}}
    config["signature"] = digest(config)
    save(directory/"manifest.json", config)
    print(directory/"manifest.json", flush=True)


def verify(config):
    if digest({k: v for k, v in config.items() if k not in {"signature", "jobs"}}) != config["signature"]:
        raise ValueError("Closed-source protocol changed")
    root = Path(config["directory"])
    for relative, checksum in config["source_hashes"].items():
        if file_hash(root/relative) != checksum:
            raise ValueError("Frozen closed-source source changed")
    for path, checksum in config["input_hashes"].items():
        if file_hash(path) != checksum:
            raise ValueError("Frozen closed-source task changed")


def load_owner(config, case):
    import torch
    for path, expected in {**config["owner_source_hashes"], **config["owner_weight_hashes"]}.items():
        if file_hash(path) != expected:
            raise ValueError("Preserved owner detector/source changed")
    sys.path.insert(0, config["owner_distseal_source"])
    sys.path.insert(0, str(Path(config["owner_distseal_source"])/"deps"))
    from distseal.loader import load_detector, parse_model_card
    detector, message = load_detector(case["detector_card"], device="cuda")
    state = torch.load(parse_model_card(case["detector_card"]).checkpoint, map_location="cpu", weights_only=True, mmap=True)["model"]
    detector.load_state_dict({k.removeprefix("detector."): v for k, v in state.items() if k.startswith("detector.")}, strict=True)
    return detector.eval().requires_grad_(False), message


def original_sanity(config, case, stop):
    import torch
    from torchvision.transforms.functional import to_tensor
    from ai_content_detector.watermarking.image import bit_scores
    task = read(case["task"])
    detector, message = load_owner(config, case)
    rows = task["images"][:2]+[row for row in task["images"] if case["mapping"][row["id"]]["cohort"] == "marked"][:2]
    tensors = torch.stack([to_tensor(read_rgb(row)) for row in rows]).cuda()
    with torch.inference_mode():
        observed = bit_scores(detector, message, tensors)
    expected = [case["mapping"][row["id"]]["original_score"] for row in rows]
    if observed != expected:
        raise ValueError(f"Owner detector does not reproduce preserved scores: {observed} != {expected}")
    save(Path(config["directory"])/case["name"]/"owner_sanity.json", {"observed": observed, "expected": expected})
    stop.check()


def owner(config, case, stop, pilot=False):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import lpips
    from pytorch_msssim import ssim
    from torchvision.transforms.functional import to_tensor
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    from ai_content_detector.watermarking.image import bit_scores
    from ai_content_detector.watermarking.protocol import detection_metrics, operating_point
    task = read(case["task"])
    sealed = verify_outputs(task, pilot)
    detector, message = load_owner(config, case)
    torch.hub.set_dir(config["metric_cache"])
    perceptual = lpips.LPIPS(net="alex", verbose=False).cuda().eval().requires_grad_(False)
    clip = CLIPModel.from_pretrained(config["clip"]["path"], local_files_only=True).cuda().eval().requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(config["clip"]["path"], local_files_only=True)
    folder = Path(config["directory"])/case["name"]
    results = {"case": case["family"], "n_per_cohort": {"marked": 32 if pilot else 512, "clean": 32 if pilot else 512, "calibration": 256},
               "access": config["access"], "conditions": {},
               "limitations": "Output purification, not modifying a closed provider. Paired metrics do not certify factual/semantic identity; 256 calibration images cannot validate extreme FPRs. No FID claim."}
    for arm in sealed:
        attack_folder = Path(task["directory"])/arm
        rows = read(attack_folder/("images_pilot.json" if pilot else "images.json"))["rows"]
        score_path = folder/(arm+"_scores.json")
        values = read(score_path) if score_path.exists() else []
        if [r["id"] for r in values] != [r["id"] for r in rows[:len(values)]]:
            raise ValueError("Owner score restart order changed")
        for start in range(len(values), len(rows), 8):
            inputs = task["images"][start:start+8]
            selected = rows[start:start+8]
            originals = [read_rgb(row) for row in inputs]
            attacked = [Image.open(attack_folder/row["file"]).convert("RGB") for row in selected]
            a = torch.stack([to_tensor(image) for image in originals]).cuda()
            b = torch.stack([to_tensor(image) for image in attacked]).cuda()
            with torch.inference_mode():
                bits = bit_scores(detector, message, b)
                lp = perceptual(a*2-1, b*2-1).flatten().cpu().tolist()
                similarity = ssim(a, b, data_range=1., size_average=False).cpu().tolist()
                mse = (a-b).square().flatten(1).mean(-1)
                psnr = (-10*mse.clamp_min(1e-12).log10()).cpu().tolist()
                pixels = processor(images=originals+attacked, return_tensors="pt").to("cuda")
                embeddings = clip.get_image_features(**pixels)
                embeddings = F.normalize(getattr(embeddings, "pooler_output", embeddings).float(), dim=-1)
                count = len(originals)
                cosines = (embeddings[:count]*embeddings[count:]).sum(-1).cpu().tolist()
            for i, row in enumerate(selected):
                values.append({"id": row["id"], **case["mapping"][row["id"]], "bit_accuracy": bits[i],
                               "lpips": lp[i], "ssim": similarity[i], "psnr": psnr[i], "clip_image_cosine": cosines[i]})
            save(score_path, values)
            stop.check()
        pools = {name: [r for r in values if r["cohort"] == name] for name in ("marked", "clean", "calibration")}
        transformed_threshold = operating_point([r["bit_accuracy"] for r in pools["calibration"]])
        conditions = {}
        for group, select in (("all", lambda r: True), ("class_zero", lambda r: r["label"] == 0), ("other_classes", lambda r: r["label"] != 0)):
            marked = [r for r in pools["marked"] if select(r)]
            null = [r for r in pools["clean"] if select(r)]
            conditions[group] = {"original_threshold": detection_metrics([r["bit_accuracy"] for r in marked], [r["bit_accuracy"] for r in null], case["original_threshold"]),
                                 "transformed_calibration_threshold": detection_metrics([r["bit_accuracy"] for r in marked], [r["bit_accuracy"] for r in null], transformed_threshold),
                                 "mean_bit_accuracy": float(np.mean([r["bit_accuracy"] for r in marked])),
                                 "quality": {key: float(np.mean([r[key] for r in marked])) for key in ("lpips", "ssim", "psnr", "clip_image_cosine")}}
        results["conditions"][arm] = conditions
        save(folder/("results_pilot.json" if pilot else "results.json"), results)
    verify_outputs(task, pilot)


def attack_worker(path, pilot=False):
    task = read(path)
    validate_task(task)
    stop = Stop()
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, stop.request)
    folder = Path(task["directory"]).parent
    name = "pilot_attack" if pilot else "attack"
    receipt = folder/(name+"_receipt.json")
    with (folder/(name+".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if receipt.exists() and read(receipt).get("status") == "completed":
            return
        save(receipt, {"status": "running", "job_id": os.environ.get("SLURM_JOB_ID")})
        try:
            attack(task, stop, pilot=pilot)
            stop.check()
            save(receipt, {"status": "completed", "job_id": os.environ.get("SLURM_JOB_ID")})
        except Pause:
            save(receipt, {"status": "paused", "signal": stop.signum})
            raise SystemExit(75 if stop.signum == signal.SIGUSR1 else 0)
        except BaseException as error:
            save(receipt, {"status": "failed", "error": repr(error)})
            raise


def phase(config, name):
    import torch
    verify(config)
    root = Path(config["directory"])
    stop = Stop()
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, stop.request)
    receipt = root/(name+"_receipt.json")
    with (root/(name+".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if receipt.exists() and read(receipt).get("status") == "completed":
            return
        save(receipt, {"status": "running", "job_id": os.environ.get("SLURM_JOB_ID")})
        try:
            if name == "preflight":
                if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 70*2**30:
                    raise ValueError("An allocated 80GB-class GPU is required")
                for case in config["cases"]:
                    attack(read(case["task"]), stop, sanity=True)
                    gc.collect(); torch.cuda.empty_cache()
                    original_sanity(config, case, stop)
                    gc.collect(); torch.cuda.empty_cache()
                timings = {}
                for case in config["cases"]:
                    task = read(case["task"])
                    records = read(Path(task["directory"])/f"sanity_{task['recipe']['primary_delta']}"/"images.json")["rows"]
                    timings[case["family"]] = {"n": len(records), "seconds_per_image": [r["seconds"] for r in records],
                                               "native_sizes": [r["size"] for r in records]}
                save(root/"preflight.json", {"gpu": torch.cuda.get_device_name(), "peak_gpu_gib": torch.cuda.max_memory_allocated()/2**30,
                                             "timings": timings, "owner_scores_reproduced": True})
            else:
                action, case_name = name.split(":", 1)
                case = next(c for c in config["cases"] if c["name"] == case_name)
                if action in ("owner", "pilot_owner"):
                    owner(config, case, stop, pilot=action == "pilot_owner")
                else:
                    raise ValueError(name)
            stop.check()
            save(receipt, {"status": "completed", "job_id": os.environ.get("SLURM_JOB_ID")})
        except Pause:
            save(receipt, {"status": "paused", "signal": stop.signum})
            raise SystemExit(75 if stop.signum == signal.SIGUSR1 else 0)
        except BaseException as error:
            save(receipt, {"status": "failed", "error": repr(error)})
            raise


def run(config, path, task):
    global CHILD
    stop = Stop()
    def forward(signum, frame):
        stop.request(signum, frame)
        if CHILD is not None:
            try:
                CHILD.send_signal(signum)
            except ProcessLookupError:
                pass
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, forward)
    verify(config)
    cases = config["cases"] if task == "lane" else [c for c in config["cases"] if c["name"] == task]
    phases = ["preflight"] if task == "preflight" else [f"{action}:{case['name']}" for case in cases for action in ("pilot_attack", "pilot_owner", "attack", "owner")]
    for name in phases:
        args = [config["python"], str(Path(config["source"])/"scripts/closed_image_watermark_experiments.py")]
        if name.startswith(("attack:", "pilot_attack:")):
            action, case_name = name.split(":", 1)
            case = next(c for c in cases if c["name"] == case_name)
            args += [action, case["task"]]
        else:
            args += ["phase", str(path), name]
        CHILD = subprocess.Popen(args)
        status = CHILD.wait()
        CHILD = None
        if stop.signum:
            raise SystemExit(75 if stop.signum == signal.SIGUSR1 else 0)
        if status:
            raise SystemExit(status)


def submit(path, preflight_only=False):
    path = Path(path).resolve()
    config = read(path)
    verify(config)
    jobs = config.get("jobs", {})
    for task in ["preflight"]+([] if preflight_only else [case["name"] for case in config["cases"]]):
        if task in jobs:
            continue
        args = ["sbatch", "--parsable", "--job-name=closed-image-"+task,
                "--output="+str(Path(config["directory"])/(task+"-%j.out"))]
        if task != "preflight":
            args.append("--dependency=afterok:"+jobs["preflight"])
        args += [str(Path(config["source"])/"scripts/run_closed_image_watermarks.slurm"), str(path), task]
        jobs[task] = subprocess.check_output(args, text=True).strip().split(";")[0]
        config["jobs"] = jobs
        save(path, config)
    print(json.dumps(jobs, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "run", "phase", "attack", "pilot_attack"))
    parser.add_argument("manifest", nargs="?")
    parser.add_argument("task", nargs="?")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "submit":
        submit(args.manifest, preflight_only=args.task == "preflight")
    elif args.action == "run":
        run(read(args.manifest), args.manifest, args.task)
    elif args.action in ("attack", "pilot_attack"):
        attack_worker(args.manifest, pilot=args.action == "pilot_attack")
    else:
        phase(read(args.manifest), args.task)
