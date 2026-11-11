"""Jina v5 text-matching evaluation of frozen original/candidate response pairs."""

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
import sys

from bira_official import save, sha
from bira_replication import Stop, Pause, digest, verify

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT/"results/research_workflow/watermarking/resources"
METHODS = ("vanilla", "bira", "direct_topk", "smoothing")


def read(path):
    return json.loads(Path(path).read_text())


def prepare(parent_path):
    parent_path = Path(parent_path).resolve()
    parent = read(parent_path)
    verify(parent)
    model = read(RESOURCES/"jina_v5_model.json")
    if model["name"] != "jinaai/jina-embeddings-v5-text-small-text-matching":
        raise ValueError("Jina v5 text-matching weights are required")
    directory = ROOT/"results/research_workflow/jina_similarity"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    source = directory/"source"
    source.mkdir(parents=True)
    shutil.copytree(ROOT/"ai_content_detector", source/"ai_content_detector", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (source/"scripts").mkdir()
    for name in ("jina_watermark_similarity.py", "run_jina_similarity.slurm", "bira_official.py", "bira_replication.py"):
        shutil.copyfile(ROOT/"scripts"/name, source/"scripts"/name)
    for case in ("textseal2", "latent"):
        (directory/case).mkdir()
    model_files = [p for p in Path(model["path"]).rglob("*") if p.is_file()]
    overlay = RESOURCES/"jina_v5_overlay"
    if not model_files or not (overlay/"transformers").is_dir():
        raise ValueError("Download Jina weights and prepare the isolated evaluation overlay first")
    config = {"directory": str(directory), "source": str(source), "parent": str(parent_path),
              "parent_signature": parent["signature"], "parent_jobs": parent["jobs"], "model": model,
              "python": parent["paper_python"], "overlay": str(overlay), "batch": 16, "dimensions": 1024,
              "seed": 20261104, "prompt": "Document: ", "max_length": 32768,
              "task": "text-matching", "pooling": "last non-padding token", "metric": "L2-normalized cosine",
              "input_hashes": {str(p): sha(p) for p in model_files},
              "frozen_hashes": {str(p.relative_to(directory)): sha(p) for p in source.rglob("*") if p.is_file()},
              "scope": "Pair original and candidate answers by prompt ID, including fresh probability continuations; cosine is not a factuality certificate."}
    config["input_hashes"].update({str(p): sha(p) for p in overlay.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc"})
    config["signature"] = digest(config)
    save(directory/"manifest.json", config)
    print(directory/"manifest.json", flush=True)


def freeze_inputs(config, case):
    from ai_content_detector.watermarking.semantic_similarity import aligned_rows
    out = Path(config["directory"])/case
    marker = out/"frozen_inputs.json"
    if marker.exists():
        frozen = read(marker)
        if frozen["signature"] != config["signature"]:
            raise ValueError("Frozen similarity inputs belong to a different protocol")
        for name, expected in frozen["hashes"].items():
            if sha(out/name) != expected:
                raise ValueError("Frozen similarity inputs changed")
        return {name: read(out/(name+".json")) for name in frozen["conditions"]}
    parent = read(config["parent"])
    if parent["signature"] != config["parent_signature"]:
        raise ValueError("Producer protocol changed")
    verify(parent)
    origin = Path(parent["directory"])/case
    complete = read(origin/"attacks_complete.json")
    for name in METHODS:
        if sha(origin/(name+".json")) != complete[name]:
            raise ValueError("Producer attack changed after owner scoring")
    original = read(origin/"inputs.json")
    pools = {"original": original, "clean_control": read(origin/"clean.json")["rows"],
             "human_control": read(origin/"humans.json")}
    pools.update({name: read(origin/(name+".json"))["rows"] for name in METHODS})
    if case == "latent":
        pools["teacher_control"] = read(origin/"teacher.json")["rows"]
    for name, rows in pools.items():
        aligned = aligned_rows(original, rows)
        if len(aligned) != 256:
            raise ValueError("Similarity evaluation requires all 256 paired test questions")
        pools[name] = [{"id": r["id"], "text": r["text"]} for r in aligned]
        save(out/(name+".json"), pools[name])
    save(marker, {"conditions": list(pools), "hashes": {name+".json": sha(out/(name+".json")) for name in pools},
                  "source_attack_hashes": complete, "signature": config["signature"]})
    return pools


def run(config, case):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from ai_content_detector.watermarking.semantic_similarity import last_token_pool, paired_scores, summarize, text_key, cosine_pairs
    verify(config)
    out = Path(config["directory"])/case
    stop = Stop()
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, stop.request)
    receipt = out/"receipt.json"
    with (out/"worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if receipt.exists() and read(receipt).get("status") == "completed":
            return
        save(receipt, {"status": "running", "job_id": os.environ.get("SLURM_JOB_ID")})
        try:
            pools = freeze_inputs(config, case)
            texts = {text_key(row["text"]): row["text"] for rows in pools.values() for row in rows}
            keys = sorted(texts)
            vectors = {}
            path = Path(config["model"]["path"])
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = None
            for start in range(0, len(keys), config["batch"]):
                batch = keys[start:start+config["batch"]]
                cache = out/f"vectors_{start:06d}.npz"
                if cache.exists():
                    with np.load(cache, allow_pickle=False) as saved:
                        if saved["keys"].tolist() != batch:
                            raise ValueError("Embedding cache input order changed")
                        values = saved["vectors"]
                else:
                    if model is None:
                        if not torch.cuda.is_available():
                            raise ValueError("Run embedding evaluation on an allocated GPU")
                        model = AutoModel.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16,
                                                          attn_implementation="sdpa").cuda().eval()
                        model.requires_grad_(False)
                    encoded = tokenizer([config["prompt"]+texts[k] for k in batch], padding=True,
                                        truncation=False, return_tensors="pt")
                    if encoded.input_ids.shape[1] > config["max_length"]:
                        raise ValueError("Jina input exceeds its context; silent truncation is forbidden")
                    encoded = encoded.to("cuda")
                    with torch.inference_mode():
                        hidden = model(**encoded, use_cache=False).last_hidden_state
                        pooled = last_token_pool(hidden, encoded.attention_mask).float()
                        values = torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()
                    del hidden, pooled, encoded
                    if values.shape != (len(batch), config["dimensions"]) or not np.isfinite(values).all():
                        raise ValueError("Invalid Jina embeddings")
                    temporary = cache.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.savez(stream, keys=np.asarray(batch), vectors=values)
                    temporary.replace(cache)
                vectors.update(zip(batch, values))
                print(case, "embedded", min(start+len(batch), len(keys)), "/", len(keys), flush=True)
                stop.check()
            results = {}
            scores = {}
            for name, rows in pools.items():
                records = paired_scores(pools["original"], rows, vectors)
                scores[name] = np.asarray([r["cosine"] for r in records])
                save(out/(name+"_similarity.json"), {"signature": config["signature"], "rows": records})
                valid = [r["cosine"] for r in records if not r["empty"]]
                results[name] = {**(summarize(valid) if valid else {"n": 0}), "empty_pairs": len(records)-len(valid)}
            originals = np.stack([vectors[text_key(r["text"])] for r in pools["original"]])
            results["mismatched_question_control"] = summarize(cosine_pairs(originals, np.roll(originals, 1, axis=0)))
            contrasts = {name: summarize(scores[name]-scores["vanilla"]) for name in METHODS if name!="vanilla"}
            summary = {"conditions": results, "paired_deltas_vs_vanilla": contrasts,
                       "model": config["model"], "task": config["task"], "dimensions": config["dimensions"],
                       "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "tokenizers")},
                       "interpretation": "Cosine measures embedding similarity, not entailment, factual accuracy, or a preservation pass rate. Fresh continuations may differ while answering the same question."}
            save(out/"summary.json", summary)
            lines = [f"# Jina v5 paired similarity: {case}", "", "| Condition | Mean cosine [95% bootstrap CI] | Median | P05 / P95 |", "|---|---:|---:|---:|"]
            for name, value in results.items():
                if value["n"]:
                    lo, hi = value["mean_ci95"]
                    lines.append(f"| {name} | {value['mean']:.4f} [{lo:.4f}, {hi:.4f}] | {value['median']:.4f} | {value['p05']:.4f} / {value['p95']:.4f} |")
            lines += ["", summary["interpretation"]]
            (out/"report.md").write_text("\n".join(lines)+"\n")
            save(receipt, {"status": "completed", "job_id": os.environ.get("SLURM_JOB_ID")})
        except Pause:
            save(receipt, {"status": "paused", "signal": stop.signum})
            raise SystemExit(75 if stop.signum == signal.SIGUSR1 else 0)
        except BaseException as error:
            save(receipt, {"status": "failed", "error": repr(error)})
            raise


def submit(path):
    path = Path(path).resolve()
    config = read(path)
    verify(config)
    jobs = config.get("jobs", {})
    for case in ("textseal2", "latent"):
        if case in jobs:
            continue
        args = ["sbatch", "--parsable", "--job-name=jina-"+case,
                "--output="+str(Path(config["directory"])/(case+"-%j.out"))]
        parent = read(config["parent"])
        producer_receipt = Path(parent["directory"])/(case+":quality_receipt.json")
        if not producer_receipt.exists() or read(producer_receipt).get("status")!="completed":
            args += ["--dependency=afterok:"+config["parent_jobs"][case]]
        args += [str(Path(config["source"])/"scripts/run_jina_similarity.slurm"), str(path), case]
        jobs[case] = subprocess.check_output(args, text=True).strip().split(";")[0]
        config["jobs"] = jobs
        save(path, config)
    print(json.dumps(jobs, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "run"))
    parser.add_argument("manifest")
    parser.add_argument("case", nargs="?", choices=("textseal2", "latent"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.manifest)
    elif args.action == "submit":
        submit(args.manifest)
    else:
        run(read(args.manifest), args.case)
