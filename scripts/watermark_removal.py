"""Prepare blind removal tasks, submit them, and evaluate frozen attack artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
from ai_content_detector.watermarking.protocol import digest, file_hash, use_sources, write_json
from ai_content_detector.watermarking.removal import artifact_hashes, validate_task, verify_artifact


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("merge", "clean_lora", "clean_full")


def read(path):
    return json.loads(Path(path).read_text())


def prepare(images_path, text_path):
    import yaml
    parents = {"images": read(images_path), "text": read(text_path)}
    directory = ROOT / "results/research_workflow/watermark_removal" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    source = directory / "source"
    shutil.copytree(ROOT / "ai_content_detector", source / "ai_content_detector",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "notebooks"))
    (source / "scripts").mkdir()
    shutil.copy2(__file__, source / "scripts/watermark_removal.py")
    upstream = Path(parents["images"]["resources"]["repositories"]["distseal"]["path"])
    native = source / "backbones/deps"
    for name in ("rar", "efficientvit"):
        shutil.copytree(upstream / "deps" / name, native / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (native / "__init__.py").touch()
    hashes = {str(p.relative_to(source)): file_hash(p) for p in source.rglob("*.py")}
    data = read(parents["images"]["data"])
    if file_hash(parents["images"]["data"]) != parents["images"]["data_sha256"]:
        raise ValueError("Original data changed")
    text_data = read(parents["text"]["data"])["text"]["removal"]
    if file_hash(parents["text"]["data"]) != parents["text"]["data_sha256"]:
        raise ValueError("Text partition changed")
    fitting = directory / "fitting"
    fitting.mkdir()
    text_fitting = {"train": [], "validation": []}
    for row in text_data:
        text_fitting["validation" if int(row["id"][:12], 16) % 5 == 0 else "train"].append(row)
    write_json(fitting / "text.json", text_fitting)
    imagenet = Path(data["images"]["source"])
    images = {"train": [], "validation": []}
    rng = np.random.default_rng(parents["images"]["seed"]+500000)
    for label, synset in enumerate(data["images"]["synsets"][:800]):
        files = sorted((imagenet / "train" / synset).glob("*.JPEG"))
        selected = rng.choice(len(files), 11, replace=False)
        for i, index in enumerate(selected):
            images["validation" if i == 10 else "train"].append({"path": str(files[index]), "label": label})
    write_json(fitting / "images.json", images)
    cards = Path(parents["images"]["resources"]["cards"])
    rar = yaml.safe_load((upstream / "configs/generation/rar.yaml").read_text())
    rar["model"]["generator"].update({"hidden_size": 1280, "num_hidden_layers": 32,
        "num_attention_heads": 16, "intermediate_size": 5120, "randomize_temperature": 1.02,
        "guidance_scale": 6.9, "guidance_scale_pow": 1.5, "use_checkpoint": False})
    uvit = yaml.safe_load((cards / "uvit-h_distseal.yaml").read_text())["model"]
    uvit.update({"input_size": 8, "pretrained_path": None, "use_checkpoint": False})
    definitions = [
        ("model01", "DistSeal-inspired Qwen3.5-4B-Base", "text", "latent_distill", "latent_clean",
         {"architecture": "causal_lm"}, str(Path(parents["text"]["directory"]) / "latent_distill/model"),
         parents["text"]["qwen"]["path"], "text.json"),
        ("model02", "DistSeal RAR-XL", "images", "rar", "rar_clean",
         {"architecture": "rar", "config": rar}, yaml.safe_load((cards / "rar_xl_distseal.yaml").read_text())["checkpoint"],
         yaml.safe_load((cards / "rar_xl_base.yaml").read_text())["checkpoint"], "images.json"),
        ("model03", "DistSeal UViT-H/DC-AE", "images", "dcae", "dcae_clean",
         {"architecture": "uvit", "config": uvit}, yaml.safe_load((cards / "uvit-h_distseal.yaml").read_text())["checkpoint"],
         yaml.safe_load((cards / "uvit-h_base.yaml").read_text())["checkpoint"], "images.json"),
    ]
    manifest = {"directory": str(directory), "source": str(source), "code_sha256": hashes,
                "parents": {"images": str(Path(images_path).resolve()), "text": str(Path(text_path).resolve())},
                "cases": {}, "jobs": {}, "qos": "qos_gpu_h100-t3", "methods": list(METHODS),
                "selection": "all predetermined candidates; no owner-score-based selection"}
    for case, label, parent_name, producer, lane, spec, supplied, clean, data_name in definitions:
        base = directory / "inputs" / case
        is_text = spec["architecture"] == "causal_lm"
        spec["checkpoint"] = str(base / "supplied" / ("model" if is_text else "model.pt"))
        if not is_text:
            spec["decoder"] = str(base / "supplied/decoder.pt")
        manifest["cases"][case] = {"label": label, "parent": parent_name, "producer": producer, "lane": lane,
            "supplied": supplied, "clean": clean, "tasks": {}, "fitting_classes": list(range(800)) if not is_text else None}
        if not is_text:
            card = "maskgit_base" if spec["architecture"] == "rar" else "dc-ae-f64c128_base"
            manifest["cases"][case]["decoder"] = yaml.safe_load((cards / f"{card}.yaml").read_text())["checkpoint"]
        for method in ("probe", *METHODS):
            recipe = ({"weights": [.75, .5, .25, 0.]} if method == "merge" else
                      {"steps": 2500, "export_steps": [500, 2500], "mode": "full" if method == "clean_full" else "lora",
                       "lr": 1e-5 if method == "clean_full" else 1e-4, "accumulation": 8 if is_text else 1,
                       "length": 512, "batch": 4, "lora_rank": 64 if is_text else 16})
            task = {"case": case, "method": method, "model": spec,
                    "fitting": {"path": str(fitting / data_name), "sha256": file_hash(fitting / data_name)},
                    "recipe": recipe, "source": str(source), "backbones": str(source / "backbones"),
                    "output": str(directory / "attacks" / case / method), "seed": parents[parent_name]["seed"],
                    "code_sha256": hashes}
            if method == "merge":
                task["clean_model"] = {"checkpoint": str(base / "clean" / ("model" if is_text else "model.pt"))}
            validate_task(task)
            path = directory / "tasks" / case / f"{method}.json"
            write_json(path, task)
            manifest["cases"][case]["tasks"][method] = {"path": str(path), "sha256": file_hash(path)}
    path = directory / "manifest.json"
    write_json(path, manifest)
    os.chmod(path, 0o600)
    report(path)
    return path


def link_or_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def release_inputs(manifest, case):
    item = manifest["cases"][case]
    parent = read(manifest["parents"][item["parent"]])
    receipt = read(Path(parent["directory"]) / f"{item['producer']}.json")
    if receipt["status"] != "completed" or receipt["job_id"] != parent["jobs"][item["producer"]]["job_id"]:
        raise ValueError("Watermarked model producer has not completed successfully")
    task = read(item["tasks"]["probe"]["path"])
    root = Path(manifest["directory"]) / "inputs" / case
    if (root / "input_seal.json").exists() and (root / "clean_seal.json").exists():
        verify_artifact(root / "supplied", read(root / "input_seal.json")["supplied"])
        verify_artifact(root / "clean", read(root / "clean_seal.json")["clean"])
        return
    for role in ("supplied", "clean"):
        destination = root / role
        if task["model"]["architecture"] == "causal_lm":
            source = Path(item[role])
            for file in source.iterdir():
                if file.is_file() and (file.suffix in {".json", ".safetensors", ".model", ".jinja"} or file.name in {"merges.txt", "vocab.txt"}):
                    if file.name != "complete.json":
                        link_or_copy(file, destination / "model" / file.name)
        else:
            link_or_copy(item[role], destination / "model.pt")
            if role == "supplied":
                link_or_copy(item["decoder"], destination / "decoder.pt")
        hashes = artifact_hashes(destination)
        if not hashes:
            raise ValueError("Model release is empty")
        write_json(root / ("input_seal.json" if role == "supplied" else "clean_seal.json"), {role: hashes})


def submit(path):
    path = Path(path).resolve()
    manifest = read(path)
    if manifest["jobs"]:
        raise ValueError("Removal run already submitted")
    for case, item in manifest["cases"].items():
        parent = read(manifest["parents"][item["parent"]])
        producer = parent["jobs"][item["producer"]]["job_id"]
        lane = parent["jobs"][item["lane"]]["job_id"]
        parent_receipt = Path(parent.get("directory", Path(manifest["parents"][item["parent"]]).parent)) / f"{item['producer']}.json"
        completed = read(parent_receipt) if parent_receipt.exists() else {}
        producer_completed = completed.get("status") == "completed" and completed.get("job_id") == producer
        predecessor = None
        for stage in ("probe", *METHODS, "evaluate"):
            hours = 1 if stage in {"probe", "merge"} else 4 if stage == "evaluate" else 20
            command = ["sbatch", "--parsable", "--hold", "--account=mbm@h100", "--partition=gpu_p6",
                       "--constraint=h100", "--gres=gpu:1", "--cpus-per-task=16", f"--qos={manifest['qos']}",
                       f"--time={hours:02}:00:00", "--signal=USR1@600", "--requeue", "--kill-on-invalid-dep=yes",
                       f"--job-name=remove-{case}-{stage}",
                       f"--output={manifest['directory']}/slurm-%j-{case}-{stage}.log"]
            if stage == "probe":
                # The completed producer is a scientific prerequisite. The old
                # removal lane only limits concurrency and may finish in any state.
                dependency = f"afterany:{lane}" if producer_completed else f"afterok:{producer},afterany:{lane}"
                command += [f"--dependency={dependency}"]
            else:
                command += [f"--dependency=afterok:{predecessor}"]
            if stage in {"probe", "evaluate"}:
                command += [str(ROOT / "scripts/run_research_slurm.sh"), "--watermark-removal-owner", str(path), case, stage]
            else:
                command += [str(ROOT / "scripts/run_research_slurm.sh"), "--watermark-removal-attack", item["tasks"][stage]["path"]]
            job = subprocess.check_output(command, text=True).strip().split(";")[0]
            manifest["jobs"][f"{case}_{stage}"] = {"job_id": job, "command": command}
            write_json(path, manifest)
            print(case, stage, job, flush=True)
            predecessor = job
    report(path)


def collect_artifacts(manifest, case):
    artifacts = []
    for method in METHODS:
        config = manifest["cases"][case]["tasks"][method]
        if file_hash(config["path"]) != config["sha256"]:
            raise ValueError("Attacker contract changed")
        task = read(config["path"])
        receipt = read(Path(task["output"]) / "receipt.json")
        if receipt["status"] != "completed" or receipt["signature"] != validate_task(task):
            raise ValueError("All attack artifacts must be frozen before owner evaluation")
        for artifact in receipt["artifacts"]:
            verify_artifact(artifact["path"], artifact["sha256"])
            artifacts.append(artifact)
    return artifacts


def evaluate_images(manifest, case, artifacts, parent, stop):
    import torch
    from distseal.loader import load_detector
    from transformers import CLIPModel, CLIPProcessor
    from ai_content_detector.watermarking.backbones import inject_lora, load_decoder, load_generator, restore_generator
    from ai_content_detector.watermarking.image import cards, sample, summarize
    from ai_content_detector.watermarking.protocol import detection_metrics
    item = manifest["cases"][case]
    task = read(item["tasks"]["probe"]["path"])
    spec = task["model"]
    family = "rar" if spec["architecture"] == "rar" else "dcae"
    baseline = read(Path(parent["directory"]) / family / "results.json")
    negatives = read(Path(parent["directory"]) / family / "base/samples.json")["rows"]
    decoder = load_decoder(spec)
    detector, message = load_detector(cards(parent, family)[2], device="cuda")
    detector.eval().requires_grad_(False)
    clip = CLIPModel.from_pretrained(parent["clip"]["path"], local_files_only=True).cuda().eval()
    processor = CLIPProcessor.from_pretrained(parent["clip"]["path"], local_files_only=True)
    output = Path(manifest["directory"]) / "evaluation" / case
    output.mkdir(parents=True, exist_ok=True)
    results = {"threshold": baseline["threshold"], "selection": manifest["selection"], "conditions": {}}
    controls = [{"name": "unmodified_supplied", "path": None, "mode": "full"}]
    for artifact in controls+artifacts:
        model = load_generator(spec)
        if artifact["path"]:
            if artifact["mode"] == "lora":
                inject_lora(model, task["recipe"]["lora_rank"])
            state = torch.load(Path(artifact["path"]) / "model.pt", map_location="cpu", weights_only=True)
            restore_generator(model, state, spec["architecture"])
            del state
        model.eval().requires_grad_(False)
        rows = sample(parent, family, model, decoder, detector, message, clip, processor,
                      output / artifact["name"], parent["image_evaluation"]["n"], stop)
        metrics = summarize(rows, negatives, baseline["threshold"], baseline["attack_thresholds"])
        for domain, predicate in (("fitting_classes", lambda y: y < 800), ("unseen_classes", lambda y: y >= 800)):
            positive = [r for r in rows if predicate(r["label"])]
            negative = [r for r in negatives if predicate(r["label"])]
            metrics[domain] = detection_metrics([r["bit_accuracy"] for r in positive],
                                                [r["bit_accuracy"] for r in negative], baseline["threshold"])
            metrics[domain]["clip_alignment"] = float(np.mean([r["clip_alignment"] for r in positive]))
        results["conditions"][artifact["name"]] = metrics
        write_json(output / "results.json", results)
        del model
        gc.collect(); torch.cuda.empty_cache()
    results["scope"] = "Fixed native generator training recipe; decoder left unchanged without consulting watermark location"
    write_json(output / "results.json", results)


def evaluate_text(manifest, case, artifacts, parent, stop):
    import torch
    from peft import PeftModel
    from ai_content_detector.watermarking.text import generate, generation_perplexity, load_model, quality
    from ai_content_detector.watermarking.latent_text import load_pair, score
    from ai_content_detector.watermarking.protocol import detection_metrics, operating_point
    item = manifest["cases"][case]
    task = read(item["tasks"]["probe"]["path"])
    data = read(parent["data"])["text"]
    output = Path(manifest["directory"]) / "evaluation" / case
    output.mkdir(parents=True, exist_ok=True)
    reference, tokenizer = load_model(parent["qwen"]["path"])
    _, extractor, _ = load_pair(parent)
    message = parent["latent_training"]["message"]
    config = {"batch": 8, "seed": parent["seed"], "temperature": .8, "top_p": 1.}
    base = generate(reference, tokenizer, data["test"], output / "base.json", config, stop)
    validation = generate(reference, tokenizer, data["validation"], output / "base_validation.json", config, stop)
    human = [tokenizer.decode(r["tokens"][50:306]) for r in data["test"]]
    human_validation = [tokenizer.decode(r["tokens"][50:306]) for r in data["validation"]]
    negative, _ = score(extractor, tokenizer, human, message)
    base_negative, _ = score(extractor, tokenizer, [r["text"] for r in base], message)
    threshold = max(operating_point(score(extractor, tokenizer, human_validation, message)[0]),
                    operating_point(score(extractor, tokenizer, [r["text"] for r in validation], message)[0]))
    results = {"threshold": threshold, "conditions": {}, "selection": manifest["selection"]}
    controls = [{"name": "unmodified_supplied", "path": task["model"]["checkpoint"], "format": "hf"}]
    for artifact in controls+artifacts:
        if artifact["format"] == "hf_adapter":
            model, _ = load_model(task["model"]["checkpoint"])
            model = PeftModel.from_pretrained(model, artifact["path"], local_files_only=True).merge_and_unload().eval()
        else:
            model, _ = load_model(artifact["path"])
        sampled = generate(model, tokenizer, data["test"], output / f"{artifact['name']}.json", config, stop)
        scores, records = score(extractor, tokenizer, [r["text"] for r in sampled], message)
        metrics = detection_metrics(scores, negative, threshold)
        metrics["against_base"] = detection_metrics(scores, base_negative, threshold)
        metrics["bit_accuracy"] = float(np.mean([r["bit_accuracy"] for r in records]))
        metrics.update(quality(model, data["test"]))
        metrics.update(generation_perplexity(reference, tokenizer, sampled))
        results["conditions"][artifact["name"]] = metrics
        write_json(output / "results.json", results)
        del model
        gc.collect(); torch.cuda.empty_cache()
    write_json(output / "results.json", results)


def run(path, case, stage):
    manifest = read(path)
    item = manifest["cases"][case]
    for name, sha in manifest["code_sha256"].items():
        if file_hash(Path(manifest["source"]) / name) != sha:
            raise ValueError("Frozen removal source changed")
    parent = read(manifest["parents"][item["parent"]])
    evaluation_receipt = Path(manifest["directory"]) / "evaluation" / case / "receipt.json"
    try:
        if stage == "probe":
            release_inputs(manifest, case)
            task = item["tasks"]["probe"]["path"]
            subprocess.run([os.sys.executable, "-m", "ai_content_detector.watermarking.removal", task], check=True)
        else:
            artifacts = collect_artifacts(manifest, case)
            write_json(evaluation_receipt, {"status": "running"})
            use_sources(parent)
            from ai_content_detector.watermarking.backbones import use_backbones
            use_backbones(Path(manifest["source"]) / "backbones")
            with StopRequest() as stop:
                if read(item["tasks"]["probe"]["path"])["model"]["architecture"] == "causal_lm":
                    evaluate_text(manifest, case, artifacts, parent, stop)
                else:
                    evaluate_images(manifest, case, artifacts, parent, stop)
            write_json(evaluation_receipt, {"status": "completed"})
    except CheckpointPause:
        if stage == "evaluate":
            write_json(evaluation_receipt, {"status": "paused"})
        subprocess.run(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]], check=True)
    except BaseException as error:
        if stage == "evaluate":
            write_json(evaluation_receipt, {"status": "failed", "error": repr(error)})
        raise
    finally:
        report(path)


def report(path):
    manifest = read(path)
    root = Path(manifest["directory"])
    lines = ["# Blind watermark removal", "", "Two families: clean-checkpoint merging and independent clean-data training "
             "(LoRA/full). All candidates are fixed before owner evaluation; no watermark score selects them.", "",
             "| Model | Stage | Job | Status |", "| --- | --- | --- | --- |"]
    for case, item in manifest["cases"].items():
        for stage in ("probe", *METHODS, "evaluate"):
            receipt = (root / "evaluation" / case / "receipt.json" if stage == "evaluate" else
                       Path(read(item["tasks"][stage]["path"])["output"]) / "receipt.json")
            status = read(receipt)["status"] if receipt.exists() else "pending"
            lines.append(f"| {item['label']} | {stage} | {manifest['jobs'].get(case+'_'+stage, {}).get('job_id', '')} | {status} |")
    lines += ["", "| Model / candidate | Owner detection | False positives | Quality |", "| --- | ---: | ---: | ---: |"]
    for case, item in manifest["cases"].items():
        path = root / "evaluation" / case / "results.json"
        if path.exists():
            for name, data in read(path)["conditions"].items():
                metrics = data.get("all", data)
                quality = f"CLIP {metrics['clip_alignment']:.3f}" if "clip_alignment" in metrics else f"PPL {metrics['human_holdout_perplexity']:.2f}"
                lines.append(f"| {item['label']} / {name} | {metrics['detection']['rate']:.1%} | {metrics['false_positive']['rate']:.1%} | {quality} |")
    textseal = read(manifest["parents"]["images"])
    control = Path(textseal["directory"]) / "textseal/results.json"
    if control.exists():
        metrics = read(control)["conditions"]["ordinary_sampler_without_original_weights"]
        lines += ["", f"TextSeal sampler-only control: ordinary inference detection {metrics['detection']['rate']:.1%}. "
                  "This is sampler bypass, not a model-weight removal result."]
    lines += ["", "Success requires reducing owner detection while preserving utility. JSON retains uncertainty and "
              "image results on fitting versus unseen classes. Complete replacement is a sanity control. "
              "Worker input separation is a research interface boundary, not an operating-system sandbox.", ""]
    (root / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "submit", "run", "report"])
    parser.add_argument("manifest")
    parser.add_argument("argument", nargs="?")
    parser.add_argument("stage", nargs="?", choices=["probe", "evaluate"])
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(args.manifest, args.argument))
    elif args.action == "submit":
        submit(args.manifest)
    elif args.action == "report":
        report(args.manifest)
    else:
        run(args.manifest, args.argument, args.stage)
