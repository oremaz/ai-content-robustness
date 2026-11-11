"""Quality-only image attack selection, followed by independent frozen owner tests."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from watermark_survival_pilot import ROOT, save, sha


def prepare(parent, quality_fallback=False, validation_seed=None, final_seed=None):
    if (validation_seed is None) != (final_seed is None):
        raise ValueError("Override both validation and final seeds together")
    if validation_seed is not None and (validation_seed < 0 or final_seed < 0 or validation_seed == final_seed):
        raise ValueError("Validation and final seeds must be distinct nonnegative integers")
    parent = Path(parent)
    attacks = json.loads(parent.read_text())
    screen = json.loads(Path(attacks["screen_manifest"]).read_text())
    protocol_path = Path(attacks["evaluation_protocol"]["path"])
    if sha(protocol_path) != attacks["evaluation_protocol"]["sha256"]:
        raise ValueError("Pre-attack evaluation protocol changed")
    protocol = json.loads(protocol_path.read_text())
    quality_fallback = quality_fallback or protocol.get("quality_fallback",False)
    directory = ROOT / "results/research_workflow/paper_writing" / (
        ("preserving-fallback-" if quality_fallback else "preserving-evaluation-")
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    source = directory / "source"
    (source / "scripts").mkdir(parents=True)
    shutil.copytree(ROOT / "ai_content_detector", source / "ai_content_detector",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.ipynb", ".pytest_cache"))
    for name in ["watermark_preserving_evaluate.py", "watermark_survival_pilot.py", "watermark_transport_train.py",
                 "run_watermark_preserving_evaluate.slurm"]:
        shutil.copyfile(ROOT / "scripts" / name, source / "scripts" / name)
    cases = {}
    for task_path in attacks["tasks"]:
        task = json.loads(Path(task_path).read_text())
        root = Path(task["directory"])
        complete = json.loads((root / "complete.json").read_text())
        if complete["signature"] != task["signature"] or complete["owner_feedback_used"]:
            raise ValueError("Blind attack contract changed")
        candidates = []
        for weight in task["recipe"]["anchor_weights"]:
            folder = root / f"anchor-{weight:g}"
            receipt = json.loads((folder / "complete.json").read_text())
            if receipt["signature"] != task["signature"] or receipt["owner_feedback_used"]:
                raise ValueError("Attack export contract changed")
            for step in task["recipe"]["export_steps"]:
                checkpoint = folder / f"step-{step}.pt"
                candidates.append({"name": f"anchor-{weight:g}-step-{step}", "anchor_weight": weight,
                                   "step": step, "checkpoint": str(checkpoint),
                                   "sha256": receipt["exports"][str(step)]})
        cases[task["name"]] = {"base": {"name": "base", "checkpoint": task["model"]["checkpoint"],
                                       "sha256": task["input_hashes"][task["model"]["checkpoint"]]},
                               "candidates": candidates, "attack_task": task_path,
                               "attack_complete_sha256": sha(root / "complete.json")}
    config = {"directory": str(directory), "source": str(source), "parent": str(parent), "protocol": protocol,
              "protocol_sha256": sha(protocol_path), "protocol_path": str(protocol_path), "cases": cases,
              "model": screen["task"]["model"], "backbones": screen["task"]["backbones"],
              "resources": screen["resources"], "clip": screen["clip"], "metric_cache": screen["metric_cache"],
              "package_map": screen["package_map"], "runtime_archive": screen["runtime_archive"],
              "clean_checkpoint": screen["clean_checkpoint"],
              "clean_checkpoint_sha256": screen["resources"]["files"]["uvit-h-in-512px-train2000k.pt"]["sha256"], "batch": 4,
              "compute": {"qos": "qos_gpu_h100-dev", "gpus_per_job": 1, "minutes": 30,
                          "maximum_allocations": 2, "validation_jobs": 3, "owner_jobs": 1},
              "input_hashes": {str(parent): sha(parent), str(protocol_path): sha(protocol_path)},
              "source_hashes": {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}}
    for filename in [config["model"]["decoder"],
                     str(Path(config["resources"]["cards"]) / "detector_dc-ae.yaml")]:
        config["input_hashes"][filename] = sha(filename)
    if quality_fallback and not protocol.get("quality_fallback"):
        protocol["quality_fallback"] = True
        protocol["validation"]["seed"] = 2026110609
        protocol["final"]["seed"] = 2026110610
        protocol["attacker_validation_minimum_passes"] = protocol["validation"]["n"]
        protocol["selection"] = ("Gate every candidate output using only the frozen quality metrics against its supplied "
            "marked source. Return the source on failure. Choose largest update budget then lowest anchor among "
            "candidates whose gated validation outputs all pass. No detector feedback. Freeze before fresh final seeds.")
        protocol["comparison"] += (" This is an output-gated hybrid client retaining the supplied source generator, "
            "not a single replaced checkpoint. It pays for both generation paths and quality evaluation. "
            "Gates are metric proxies, not proof of exact perceptual or semantic equivalence.")
        protocol["status"] = ("New development protocol frozen after the earlier selected25-update experiment showed "
            "zero removals for the original model and both candidates. Existing training exports and class identities "
            "are reused; validation/final generation seeds are fresh. Further confirmation is required.")
    if validation_seed is not None:
        protocol["validation"]["seed"] = validation_seed
        protocol["final"]["seed"] = final_seed
    if quality_fallback or validation_seed is not None:
        protocol_path = directory/"evaluation_protocol.json"
        save(protocol_path, protocol)
        config["protocol_path"], config["protocol_sha256"] = str(protocol_path), sha(protocol_path)
        config["input_hashes"][str(protocol_path)] = sha(protocol_path)
    config["signature"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    save(directory / "manifest.json", config)
    print(directory / "manifest.json")


def run(path, stage, case=None):
    print("image evaluation starting", stage, case, flush=True)
    import numpy as np
    from PIL import Image
    import torch
    from unittest.mock import patch
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
    from ai_content_detector.rl_evasion.config import seed_everything
    from ai_content_detector.watermarking.backbones import use_backbones, load_generator, load_decoder, restore_generator
    from ai_content_detector.watermarking.image_quality import ImageQuality, read_pair
    from ai_content_detector.watermarking.preservation import numeric_gate, select_preserving_candidate, joint_outcomes, preserving_fallback

    config = json.loads(Path(path).read_text())
    root = Path(config["directory"])
    for filename, expected in config["input_hashes"].items():
        if sha(filename) != expected:raise ValueError("Evaluation input changed")
    for relative, expected in config["source_hashes"].items():
        if sha(Path(config["source"]) / relative) != expected:raise ValueError("Evaluation source changed")
    use_backbones(config["backbones"])
    sys.path.insert(0, config["resources"]["repositories"]["distseal"]["path"])
    with patch("importlib.metadata.packages_distributions", return_value=config["package_map"]):
        from distseal.loader import load_detector
    save(root / (f"environment-{stage}-{case or 'all'}.json"), {"job_id": os.environ.get("SLURM_JOB_ID"),
         "python": sys.version, "torch": torch.__version__, "gpu": torch.cuda.get_device_name()})
    protocol = config["protocol"]
    selection_hashes = {}
    if stage == "owner":
        for name in config["cases"]:
            complete = json.loads((root / name / "validation_complete.json").read_text())
            selection_path = root / name / "selection.json"
            if complete["signature"] != config["signature"] or sha(selection_path) != complete["selection_sha256"]:
                raise ValueError("Blind quality selection changed")
            if json.loads(selection_path.read_text())["owner_scores_used"] is not False:
                raise ValueError("Owner feedback entered quality selection")
            selection_hashes[name] = sha(selection_path)
        save(root / "owner_input_freeze.json", {"signature": config["signature"], "selection_hashes": selection_hashes,
                                               "protocol_sha256": config["protocol_sha256"]})
    elif case not in config["cases"]:
        raise ValueError("Validation requires one declared checkpoint case")
    decoder = load_decoder(config["model"])
    model = load_generator(config["model"])
    with patch("importlib.metadata.packages_distributions", return_value=config["package_map"]):
        metrics = ImageQuality(config["clip"]["path"], config["metric_cache"])

    with StopRequest() as stop:
        def pause():
            if stop.requested:raise CheckpointPause("Image evaluation batch saved")

        def generate(candidate, folder, design):
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "samples.json"
            cached = json.loads(path.read_text()) if path.exists() else None
            if cached and (cached["signature"] != config["signature"] or cached["checkpoint_sha256"] != candidate["sha256"]):
                raise ValueError("Generation restart protocol changed")
            rows = cached["rows"] if cached else []
            if len(rows) > design["n"] or [r["index"] for r in rows] != list(range(len(rows))):
                raise ValueError("Generation restart indices changed")
            if len(rows) < design["n"]:
                if sha(candidate["checkpoint"]) != candidate["sha256"]:raise ValueError("Frozen generator checkpoint changed")
                state = torch.load(candidate["checkpoint"], map_location="cpu", weights_only=True)
                restore_generator(model, state, "uvit")
                del state
            for start in range(len(rows), design["n"], config["batch"]):
                seed_everything(design["seed"] + 10000 + start)
                labels = torch.tensor(design["labels"][start:start+config["batch"]], device="cuda", dtype=torch.int32)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model.generate(labels, torch.full_like(labels, 1000), scale=1.)
                    pixels = decoder.decode(z / .2889).float().mul(127.5).add(128).clamp(0, 255).to(torch.uint8)
                for offset, image in enumerate(pixels):
                    index = start + offset
                    filename = folder / f"{index:05}.png"
                    Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(filename)
                    rows.append({"index": index, "label": design["labels"][index], "path": str(filename), "sha256": sha(filename)})
                save(path, {"signature": config["signature"], "checkpoint_sha256": candidate["sha256"], "rows": rows})
                print(stage, case, folder.name, "samples", len(rows), flush=True)
                pause()
            return rows

        def quality(originals, candidates, folder):
            path = folder / "quality.json"
            rows = json.loads(path.read_text()) if path.exists() else []
            if len(originals) != len(candidates) or [r["index"] for r in rows] != list(range(len(rows))):
                raise ValueError("Quality sample alignment changed")
            for start in range(len(rows), len(candidates), config["batch"]):
                pairs, labels = [], []
                for a, b in zip(originals[start:start+config["batch"]], candidates[start:start+config["batch"]]):
                    if a["index"] != b["index"] or a["label"] != b["label"]:raise ValueError("Paired image labels changed")
                    pairs.append(read_pair({"index": a["index"], "reference": a["path"], "candidate": b["path"],
                                            "reference_sha256": a["sha256"], "candidate_sha256": b["sha256"]}))
                    labels.append(a["label"])
                values = metrics.score([p[0] for p in pairs], [p[1] for p in pairs], labels)
                for offset, value in enumerate(values):
                    reference = value["reference_clip_alignment"]
                    value["class_alignment_retention"] = value["clip_alignment"] / reference if reference > 0 else None
                    rows.append({"index": start + offset, **value, "preservation": numeric_gate(value, protocol["quality_rule"])})
                save(path, rows)
                pause()
            return rows

        def gated(originals, candidates, values, folder):
            selected, fallback_ids = preserving_fallback(
                {str(r["index"]): r for r in originals}, {str(r["index"]): r for r in candidates},
                {str(r["index"]): r["preservation"] for r in values})
            rows = [selected[str(i)] for i in range(len(originals))]
            folder.mkdir(parents=True, exist_ok=True)
            save(folder/"samples.json", {"signature": config["signature"], "rows": rows,
                                        "fallback_ids": fallback_ids, "owner_scores_used": False})
            return rows, quality(originals, rows, folder)

        if stage == "validate":
            data = config["cases"][case]
            folder = root / case
            baseline = generate(data["base"], folder / "validation-base", protocol["validation"])
            summaries = []
            for candidate in data["candidates"]:
                destination = folder / candidate["name"]
                images = generate(candidate, destination, protocol["validation"])
                values = quality(baseline, images, destination)
                raw_preserved = sum(r["preservation"]["passed"] for r in values)
                if protocol.get("quality_fallback"):
                    _, values = gated(baseline, images, values, destination/"gated")
                summaries.append({**candidate, "n": len(values),
                                  "preserved": sum(r["preservation"]["passed"] for r in values),
                                  "raw_preserved": raw_preserved,
                                  "quality_sha256": sha(destination / "quality.json")})
                print(case, candidate["name"], "preserved", summaries[-1]["preserved"], flush=True)
            selected = select_preserving_candidate(summaries, protocol["attacker_validation_minimum_passes"])
            save(folder / "selection.json", {"signature": config["signature"], "owner_scores_used": False,
                "selected": selected or data["base"], "nonzero_attack_eligible": selected is not None,
                "candidates": summaries, "policy": protocol["selection"]})
            save(folder / "validation_complete.json", {"signature": config["signature"],
                                                       "selection_sha256": sha(folder / "selection.json")})
            return

        pools, quality_rows, raw_quality_rows = {}, {}, {}
        for name, data in config["cases"].items():
            choice = json.loads((root / name / "selection.json").read_text())["selected"]
            baseline = generate(data["base"], root / name / "final-base", protocol["final"])
            attacked = generate(choice, root / name / "final-attack", protocol["final"])
            quality_rows[name] = quality(baseline, attacked, root / name / "final-attack")
            if protocol.get("quality_fallback"):
                raw_quality_rows[name] = quality_rows[name]
                pools[name+":raw_attack"] = attacked
                attacked, quality_rows[name] = gated(baseline, attacked, quality_rows[name], root/name/"final-gated")
            pools[name + ":base"], pools[name + ":attack"] = baseline, attacked
        null = {"checkpoint": config["clean_checkpoint"], "sha256": config["clean_checkpoint_sha256"]}
        pools["clean_null"] = generate(null, root / "final-clean-null", protocol["final"])
        save(root / "final_artifact_freeze.json", {"signature": config["signature"],
            "selection_hashes": selection_hashes, "samples": pools,
            "quality_hashes": {name: sha(root/name/("final-gated" if protocol.get("quality_fallback") else "final-attack")/"quality.json") for name in config["cases"]},
            "raw_quality_hashes": {name: sha(root/name/"final-attack/quality.json") for name in raw_quality_rows}})
        del metrics, model, decoder
        torch.cuda.empty_cache()
        from ai_content_detector.watermarking.image import bit_scores, cards
        detector, message = load_detector(cards(config, "dcae")[2], device="cuda")
        detector.eval().requires_grad_(False)
        decisions, sample_scores = {}, {}
        for name, images in pools.items():
            scores = []
            for start in range(0, len(images), config["batch"]):
                arrays = []
                for row in images[start:start+config["batch"]]:
                    if sha(row["path"]) != row["sha256"]:raise ValueError("Frozen final PNG changed")
                    with Image.open(row["path"]) as image:arrays.append(np.asarray(image.convert("RGB"), dtype=np.float32)/255)
                tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).cuda()
                with torch.no_grad():scores.extend(bit_scores(detector, message, tensor))
            sample_scores[name] = scores
            decisions[name] = {str(i): bool(score > protocol["owner_threshold"]) for i, score in enumerate(scores)}
        results = {}
        for name in config["cases"]:
            gates = {str(r["index"]): r["preservation"] for r in quality_rows[name]}
            results[name] = joint_outcomes(decisions[name+":base"], decisions[name+":attack"], gates)
        save(root / "owner_results.json", {"signature": config["signature"], "conditions": results,
            "raw_conditions": {name: joint_outcomes(decisions[name+":base"], decisions[name+":raw_attack"],
                 {str(r["index"]): r["preservation"] for r in values}) for name, values in raw_quality_rows.items()},
            "owner_threshold": protocol["owner_threshold"], "selection_hashes": selection_hashes,
            "bit_accuracies": sample_scores, "null_detected": sum(decisions["clean_null"].values()),
            "null_n": len(decisions["clean_null"]), "scope": "Development final test;64nullimagescannotpreciselyvalidate1percentFPR. Multi-seed confirmation and additional watermark families remain required."})
        print(json.dumps(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "validate", "owner"])
    parser.add_argument("manifest")
    parser.add_argument("case", nargs="?", choices=["original", "fixed", "adapted"])
    parser.add_argument("--quality-fallback", action="store_true")
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--final-seed", type=int)
    args = parser.parse_args()
    if args.action == "prepare":prepare(args.manifest, args.quality_fallback, args.validation_seed, args.final_seed)
    else:
        from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
        try:run(args.manifest, args.action, args.case)
        except CheckpointPause as error:
            print(str(error), flush=True)
            sys.exit(75)
