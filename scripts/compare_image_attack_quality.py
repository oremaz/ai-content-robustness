"""Extend saved weight-attack and NFPA evaluations with the same quality metrics."""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

from bira_official import save, sha
from bira_replication import Stop, Pause, digest, verify

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT/"results/research_workflow/watermarking/20261026T001727478096Z"
WEIGHTS = ROOT/"results/research_workflow/watermark_removal/20261026T110140724565Z"
NFPA = ROOT/"results/research_workflow/closed_image/20261104T082631753968Z"


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    from ai_content_detector.watermarking.image_quality import align_by_index
    nfpa = read(NFPA/"manifest.json")
    directory = ROOT/"results/research_workflow/image_quality"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    source = directory/"source"
    source.mkdir(parents=True)
    shutil.copytree(ROOT/"ai_content_detector", source/"ai_content_detector", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (source/"scripts").mkdir()
    for name in ("compare_image_attack_quality.py", "run_image_attack_quality.slurm", "bira_official.py", "bira_replication.py"):
        shutil.copyfile(ROOT/"scripts"/name, source/"scripts"/name)
    inputs = {}
    cases = {}
    for family, weight_case, nfpa_case in (("rar", "model02", "model01"), ("dcae", "model03", "model02")):
        old = WEIGHTS/"evaluation"/weight_case
        conditions = read(old/"results.json")["conditions"]
        inputs[str(old/"results.json")] = sha(old/"results.json")
        reference_dir = ORIGINAL/family/"watermarked"
        reference = sorted(read(reference_dir/"samples.json")["rows"], key=lambda r:r["index"])
        if len(reference) != 512:
            raise ValueError("Expected 512 preserved watermarked reference images")
        references = {r["index"]: {"path": str(reference_dir/f"{r['index']:05}.png"),
                                   "sha256": sha(reference_dir/f"{r['index']:05}.png"), "label": r["label"]} for r in reference}
        by_hash = {r["sha256"]: index for index, r in references.items()}
        if len(by_hash) != len(reference):
            raise ValueError("Ambiguous reference PNG hashes")
        pools = {}
        for name, metrics in conditions.items():
            origin = old/name
            aligned = align_by_index(reference, read(origin/"samples.json")["rows"])
            inputs[str(origin/"samples.json")] = sha(origin/"samples.json")
            records = []
            for row in aligned:
                index = row["index"]; ref = references[index]; candidate = origin/f"{index:05}.png"
                checksum = sha(candidate)
                if name == "unmodified_supplied" and checksum != ref["sha256"]:
                    raise ValueError("Earlier baseline is not identical to the NFPA reference")
                records.append({"index": index, "label": row["label"], "reference": ref["path"],
                                "reference_sha256": ref["sha256"], "candidate": str(candidate),
                                "candidate_sha256": checksum, "prior": {"clip_alignment": row["clip_alignment"]}})
            pools["weight_"+name] = {"rows": records, "detection": metrics["all"]["detection"], "access": "weights"}
        case = next(c for c in nfpa["cases"] if c["name"] == nfpa_case)
        task = read(case["task"])
        input_rows = {r["id"]: r for r in task["images"]}
        scores_dir = NFPA/nfpa_case
        nfpa_results = read(scores_dir/"results.json")["conditions"]
        inputs[str(scores_dir/"results.json")] = sha(scores_dir/"results.json")
        for arm, metrics in nfpa_results.items():
            path = scores_dir/(arm+"_scores.json")
            inputs[str(path)] = sha(path)
            index_path = Path(task["directory"])/arm/"images.json"
            generated = {r["id"]: r for r in read(index_path)["rows"]}
            records = []
            for row in read(path):
                if row["cohort"] != "marked":
                    continue
                original = input_rows[row["id"]]
                index = by_hash.get(original["sha256"])
                if index is None or references[index]["label"] != row["label"]:
                    raise ValueError("NFPA original cannot be matched to reference by PNG hash/class")
                candidate = index_path.parent/generated[row["id"]]["file"]
                checksum = sha(candidate)
                if checksum != generated[row["id"]]["sha256"]:
                    raise ValueError("NFPA candidate differs from its sealed artifact")
                ref = references[index]
                records.append({"index": index, "label": row["label"], "reference": ref["path"],
                                "reference_sha256": ref["sha256"], "candidate": str(candidate), "candidate_sha256": checksum,
                                "prior": {k: row[k] for k in ("lpips", "ssim", "psnr", "clip_image_cosine")}})
            records = align_by_index(reference, records)
            pools["output_"+arm] = {"rows": records, "detection": metrics["all"]["original_threshold"]["detection"], "access": "output images"}
        (directory/family).mkdir()
        save(directory/family/"pairs.json", pools)
        inputs[str(directory/family/"pairs.json")] = sha(directory/family/"pairs.json")
        cases[family] = str(directory/family/"pairs.json")
        print(family, len(pools), "conditions, identical references verified", flush=True)
    for base in (Path(nfpa["clip"]["path"]), Path(nfpa["metric_cache"])/"checkpoints"):
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in (".json", ".bin", ".safetensors", ".pth"):
                inputs[str(p)] = sha(p)
    config = {"directory": str(directory), "source": str(source), "cases": cases, "clip": nfpa["clip"],
              "metric_cache": nfpa["metric_cache"], "batch": 8, "seed": 20261104,
              "pairing": "All 512 original PNGs match bit-for-bit across the weight and NFPA baselines. Weight candidates share original class/index/generation seed; NFPA pairs are identified by original PNG hash.",
              "metrics": "LPIPS-Alex v0.1, native-size SSIM, PSNR capped at 120dB, CLIP image-image cosine and original ImageNet class-text cosine.",
              "input_hashes": inputs,
              "frozen_hashes": {str(p.relative_to(directory)): sha(p) for p in source.rglob("*") if p.is_file()}}
    config["signature"] = digest(config)
    save(directory/"manifest.json", config)
    print(directory/"manifest.json", flush=True)


def report(config, family, pools, outputs):
    import numpy as np
    from ai_content_detector.watermarking.image_quality import METRICS
    from ai_content_detector.watermarking.semantic_similarity import summarize
    results = {}
    for name, rows in outputs.items():
        groups = {}
        for group, select in (("all", lambda r:True), ("class_zero", lambda r:r["label"]==0),
                              ("other_classes", lambda r:r["label"]!=0), ("unseen_classes", lambda r:r["label"]>=800)):
            chosen = [r for r in rows if select(r)]
            groups[group] = {key:summarize([r[key] for r in chosen]) for key in METRICS} if chosen else {}
        results[name] = {"access": pools[name]["access"], "detection": pools[name]["detection"], "quality": groups,
                         "mean_clip_alignment_delta": float(np.mean([r["clip_alignment"]-r["reference_clip_alignment"] for r in rows]))}
    folder = Path(config["directory"])/family
    save(folder/"results.json", {"conditions": results, "pairing": config["pairing"],
         "versions": {key:importlib.metadata.version(key) for key in ("torch", "transformers", "lpips", "pytorch-msssim")},
         "interpretation": "Paired pixel/perceptual distances measure sample drift. Weight-modified generators may produce different valid images even for the same class/seed; low similarity alone is not poor generation quality. CLIP class alignment is the common semantic proxy. No metric certifies factual or identity preservation."})
    lines = [f"# Matched image quality: {family}", "", "All conditions use the same 512 reference images.", "",
             "| Condition | Detected | LPIPS ↓ | SSIM ↑ | PSNR ↑ | Image CLIP ↑ | Class CLIP ↑ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in results.items():
        q = value["quality"]["all"]
        cells = [f"{100*value['detection']['rate']:.1f}%"]+[f"{q[key]['mean']:.4f}" for key in METRICS if key!="reference_clip_alignment"]
        lines.append("| "+name+" | "+" | ".join(cells)+" |")
    lines += ["", "Paired distances measure sample drift; class alignment measures agreement with the requested class. Weight changes can produce a different valid image. These metrics do not define a universal quality pass threshold."]
    (folder/"report.md").write_text("\n".join(lines)+"\n")


def run(config, family):
    import numpy as np
    import torch
    from ai_content_detector.watermarking.image_quality import ImageQuality, read_pair, METRICS
    verify(config)
    out = Path(config["directory"])/family
    stop = Stop()
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, stop.request)
    receipt = out/"receipt.json"
    with (out/"worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if receipt.exists() and read(receipt).get("status")=="completed":
            return
        save(receipt, {"status":"running", "job_id":os.environ.get("SLURM_JOB_ID")})
        try:
            if not torch.cuda.is_available():
                raise ValueError("Allocate a GPU for the image quality evaluation")
            engine = ImageQuality(config["clip"]["path"], config["metric_cache"])
            pools = read(config["cases"][family]); outputs = {}; prior_differences = {}
            for name, pool in pools.items():
                rows = pool["rows"]
                path = out/(name+".json")
                stored = read(path) if path.exists() else {"signature":config["signature"], "rows":[]}
                done = stored["rows"]
                if stored["signature"]!=config["signature"] or [r["index"] for r in done]!=[r["index"] for r in rows[:len(done)]]:
                    raise ValueError("Image quality resume inputs changed")
                for start in range(len(done), len(rows), config["batch"]):
                    batch = rows[start:start+config["batch"]]
                    pairs = [read_pair(r) for r in batch]
                    metrics = engine.score([p[0] for p in pairs], [p[1] for p in pairs], [r["label"] for r in batch])
                    for row, scores in zip(batch, metrics):
                        if not all(np.isfinite(scores[key]) for key in METRICS):
                            raise ValueError("Nonfinite image quality metric")
                        if any(abs(scores[key]-value)>1e-3 for key,value in row["prior"].items()):
                            raise ValueError(f"Existing metric was not reproduced for {name}/{row['index']}")
                        done.append({"index":row["index"], "label":row["label"], **scores})
                    save(path, stored)
                    stop.check()
                outputs[name] = done
                prior_differences[name] = {key:max(abs(a[key]-b["prior"][key]) for a,b in zip(done,rows)) for key in rows[0]["prior"]}
                print(family, name, len(done), "metrics completed", flush=True)
            report(config, family, pools, outputs)
            save(out/"previous_metric_agreement.json", prior_differences)
            save(receipt, {"status":"completed", "job_id":os.environ.get("SLURM_JOB_ID")})
        except Pause:
            save(receipt, {"status":"paused", "signal":stop.signum})
            raise SystemExit(75 if stop.signum==signal.SIGUSR1 else 0)
        except BaseException as error:
            save(receipt, {"status":"failed", "error":repr(error)})
            raise


def submit(path):
    path = Path(path).resolve();config = read(path);verify(config);jobs = config.get("jobs", {})
    for family in config["cases"]:
        if family in jobs:
            continue
        args = ["sbatch", "--parsable", "--job-name=quality-"+family,
                "--output="+str(Path(config["directory"])/(family+"-%j.out")),
                str(Path(config["source"])/"scripts/run_image_attack_quality.slurm"), str(path), family]
        jobs[family] = subprocess.check_output(args, text=True).strip().split(";")[0]
        config["jobs"] = jobs;save(path,config)
    print(json.dumps(jobs,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("prepare","submit","run"));p.add_argument("manifest",nargs="?");p.add_argument("family",nargs="?")
    args=p.parse_args()
    if args.action=="prepare":prepare()
    elif args.action=="submit":submit(args.manifest)
    else:run(read(args.manifest),args.family)
