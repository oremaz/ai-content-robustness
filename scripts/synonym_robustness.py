"""Checkpointed synonym sensitivity test on saved base-Qwen outputs."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

from scripts.research_chain import ROOT, OUTPUT, write_record

MODEL = "google/gemma-1.1-2b-it"
REVISION = "d750f5eceb83e978c09e2b3597c2a8784e381022"


def inputs():
    from scripts.architecture_races import inputs as race_inputs
    from ai_content_detector.rl_evasion.defender_comparison import workflow

    identity, plan = race_inputs()
    identity["initial_defenders"] = workflow.checkpoint_identity(workflow.warm_checkpoints())
    baseline = workflow.OUTPUT / "warmup/measurements.json"
    identity["warmup_measurements_sha256"] = hashlib.sha256(baseline.read_bytes()).hexdigest()
    identity["wordnet_sha256"] = hashlib.sha256((ROOT / ".venv/nltk_data/corpora/wordnet.zip").read_bytes()).hexdigest()
    return identity, plan


def submit():
    from scripts.short_probes import split_audit

    identity, plan = inputs()
    races = json.loads((OUTPUT / "architecture_races.json").read_text())
    path = OUTPUT / "defender_comparison/synonym_robustness" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") / "manifest.json"
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "inputs": identity, "baseline_plan": plan,
                "split_audit": split_audit(), "rate": 0.05, "seed": 42, "model": MODEL, "revision": REVISION,
                "max_new_tokens": 256, "detector_feedback": False, "training_updates": 0,
                "source": "saved_base_Qwen_evaluation_outputs", "human_control": "original_fixed_human_passages",
                "held": True, "max_restarts": 100, "capacity_race": races["jobs"]["roberta"]["job_id"]}
    write_record(path, manifest)
    command = ["sbatch", "--parsable", "--hold", "--qos=qos_gpu_h100-dev", "--time=02:00:00",
               "--signal=USR1@600", "--requeue", "--cpus-per-task=32", "--job-name=synonym-robustness",
               f"--output={path.parent}/slurm-%j.log", str(ROOT / "scripts/run_research_slurm.sh"),
               "--synonym-robustness", str(path)]
    job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";", 1)[0]
    if not job.isdigit():
        raise RuntimeError(f"Unexpected Slurm response: {job!r}")
    manifest.update(job_id=job, command=command)
    write_record(path, manifest)
    print(f"Job {job}: {path}", flush=True)
    return path


def release(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if not manifest["held"]:
        raise ValueError("Synonym job is already released")
    race = manifest["capacity_race"]

    def state():
        return subprocess.check_output(["squeue", "--noheader", "--jobs", race, "--format=%T"], text=True).strip()

    if state() != "PENDING":
        raise RuntimeError("Capacity race is no longer pending; inspect GPU usage before release")
    subprocess.run(["scontrol", "hold", race], check=True)
    try:
        if state() != "PENDING":
            raise RuntimeError("Capacity race started during handover; synonym job remains held")
        dependency = f"afterany:{manifest.get('capacity_blocker', manifest['job_id'])}"
        subprocess.run(["scontrol", "update", f"JobId={race}", f"Dependency={dependency}"], check=True)
        manifest["capacity_dependency"] = dependency
        write_record(path, manifest)
        subprocess.run(["scontrol", "release", manifest["job_id"]], check=True)
        manifest.update(held=False, released_at=datetime.now(timezone.utc).isoformat())
        write_record(path, manifest)
    finally:
        subprocess.run(["scontrol", "release", race], check=True)


def variants(manifest, directory, pairs, stop):
    import gc
    import nltk
    from nltk.corpus import wordnet
    import spacy
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ai_content_detector.rl_evasion.benchmarking.synonym_robustness import candidates, request_text, apply_proposal
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
    from ai_content_detector.rl_evasion.defender_comparison.protocol import digest

    signature = digest({"inputs": manifest["inputs"], "rate": manifest["rate"], "seed": manifest["seed"],
                        "model": manifest["model"], "revision": manifest["revision"],
                        "max_new_tokens": manifest["max_new_tokens"],
                        **({"target_budget": True} if manifest.get("target_budget") else {})})
    nltk.data.path.insert(0, str(ROOT / ".venv/nltk_data"))
    wordnet.ensure_loaded()
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    model, tokenizer, records = None, None, []
    try:
        for index, pair in enumerate(pairs):
            if stop.requested:
                raise CheckpointPause("Synonym edit proposals retained")
            path = directory / "variants" / f"{index:04d}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["signature"] != signature or record["original"] != pair["ai"]:
                    raise ValueError("Saved synonym proposal differs from the immutable inputs")
            else:
                specification = candidates(nlp(pair["ai"]), wordnet, index, manifest["rate"], manifest["seed"])
                response = '{"edits":[]}'
                if specification["budget"] and specification["candidates"]:
                    if model is None:
                        tokenizer = AutoTokenizer.from_pretrained(manifest["model"], revision=manifest["revision"], local_files_only=True)
                        model = AutoModelForCausalLM.from_pretrained(manifest["model"], revision=manifest["revision"],
                                                                   local_files_only=True, torch_dtype=torch.bfloat16,
                                                                   device_map="cuda:0").eval().requires_grad_(False)
                    message = request_text(pair["ai"], specification, target_budget=manifest.get("target_budget", False))
                    encoded = tokenizer.apply_chat_template([{"role": "user", "content": message}],
                                                             add_generation_prompt=True, tokenize=True,
                                                             return_dict=True, return_tensors="pt").to("cuda")
                    if encoded["input_ids"].shape[1] + manifest["max_new_tokens"] > model.config.max_position_embeddings:
                        raise ValueError("Synonym proposal exceeds the instruction model context window")
                    with torch.inference_mode():
                        generated = model.generate(**encoded, do_sample=False, max_new_tokens=manifest["max_new_tokens"],
                                                   pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                                                                 else tokenizer.eos_token_id))
                    response = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
                record = {"index": index, "sample_id": pair["sample_id"], "signature": signature,
                          "original": pair["ai"], **specification, "response": response,
                          **apply_proposal(pair["ai"], specification, response)}
                write_record(path, record)
                print(f"Synonym row {index + 1}/{len(pairs)}: {len(record['edits'])}/{record['budget']} edits", flush=True)
            records.append(record)
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    write_record(directory / "variants.json", records)
    return records


def experiment(manifest, directory, stop):
    import numpy as np
    from ai_content_detector.rl_evasion.benchmarking.synonym_robustness import paired_metrics
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
    from ai_content_detector.rl_evasion.defender_comparison import workflow
    from ai_content_detector.rl_evasion.defender_comparison.defenders import DefenderPool
    from ai_content_detector.rl_evasion.defender_comparison.protocol import digest, score_checkpointed
    from ai_content_detector.rl_evasion.text_evasion.rewards import SemanticSimilarityReward

    identity, plan = inputs()
    if identity != manifest["inputs"] or plan != manifest["baseline_plan"]:
        raise ValueError("Synonym inputs or frozen warmed detectors changed since submission")
    if manifest["rate"] != 0.05 or manifest["training_updates"] != 0 or manifest["detector_feedback"]:
        raise ValueError("Synonym experiment differs from the agreed detector-blind 5% protocol")
    result_path = directory / "results.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    pairs = json.loads((workflow.OUTPUT / "base_pairs.json").read_text())["evaluation"]
    baseline = json.loads((workflow.OUTPUT / "warmup/measurements.json").read_text())
    if len(pairs) != 500 or not all(row["eligible"] for row in pairs):
        raise ValueError("The 500 recorded paired base-Qwen outputs must all be eligible")
    if baseline["eligible_ids"] != [row["sample_id"] for row in pairs]:
        raise ValueError("Cached warmed baseline scores are not aligned with the fixed evaluation rows")
    records = variants(manifest, directory, pairs, stop)
    return measure_variants(manifest, directory, pairs, baseline, plan, records, stop)


def measure_variants(manifest, directory, pairs, baseline, plan, records, stop):
    import numpy as np
    from ai_content_detector.rl_evasion.benchmarking.synonym_robustness import paired_metrics
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
    from ai_content_detector.rl_evasion.defender_comparison import workflow
    from ai_content_detector.rl_evasion.defender_comparison.defenders import DefenderPool
    from ai_content_detector.rl_evasion.defender_comparison.protocol import digest, score_checkpointed
    from ai_content_detector.rl_evasion.text_evasion.rewards import SemanticSimilarityReward

    result_path = directory / "results.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    indices = manifest.get("sample_indices", list(range(len(pairs))))
    baseline_count = len(baseline["eligible_ids"])
    if (len(records) != len(pairs) or len(indices) != len(pairs)
            or [baseline["eligible_ids"][index] for index in indices] != [row["sample_id"] for row in pairs]):
        raise ValueError("Selected rows are not aligned with the original paired baseline")
    changed = [bool(row["edits"]) for row in records]
    if not any(changed):
        raise RuntimeError("No synonym edits were accepted; no robustness conclusion can be measured")
    frozen = digest(records)
    write_record(directory / "frozen_variants.json", {"sha256": frozen, "rows": len(records),
                                                    "detector_feedback": False})
    semantic_path = directory / "semantic.json"
    semantic = json.loads(semantic_path.read_text()) if semantic_path.exists() else {"variants_sha256": frozen, "values": []}
    if semantic["variants_sha256"] != frozen:
        raise ValueError("Synonym semantic cache refers to different rewrites")
    encoder = SemanticSimilarityReward(device="cpu")
    for row in records[len(semantic["values"]):]:
        if stop.requested:
            raise CheckpointPause("Synonym semantic diagnostics retained")
        semantic["values"].append(encoder(row["original"], row["text"]) if row["edits"] else 1.0)
        if not np.isfinite(semantic["values"][-1]):
            raise ValueError("Nonfinite original-versus-rewritten semantic similarity")
        write_record(semantic_path, semantic)
    del encoder
    if stop.requested:
        raise CheckpointPause("Synonym diagnostics retained before detector loading")
    names = manifest.get("detector_names", list(plan["specifications"]))
    pool = DefenderPool({name: plan["specifications"][name] for name in names}, directory / "defenders",
                        checkpoints={name: value for name, value in workflow.warm_checkpoints().items() if name in names},
                        retain_on_cpu=False)
    try:
        scores = score_checkpointed(pool, [row["text"] for row in records], directory / "scores")
    finally:
        pool.clear()
    before = {name: [baseline["scores"][name][index] for index in indices] for name in names}
    human = {name: [baseline["scores"][name][baseline_count + index] for index in indices] for name in names}
    if set(scores) != set(before):
        raise ValueError("All requested warmed detectors must complete synonym scoring")
    if manifest.get("include_ensemble", True):
        before["ensemble_mean"] = np.mean(list(before.values()), axis=0).tolist()
        human["ensemble_mean"] = np.mean(list(human.values()), axis=0).tolist()
        scores["ensemble_mean"] = np.mean(list(scores.values()), axis=0).tolist()
    metrics = [{"detector": name, **paired_metrics(before[name], scores[name], human[name], changed)} for name in before]
    rows = [{"detector": name, "index": index, "sample_id": record["sample_id"],
             "original_score": before[name][index], "synonym_score": scores[name][index],
             "score_delta": scores[name][index] - before[name][index],
             "accepted_edits": len(record["edits"]), "eligible_content_words": record["eligible_content_words"],
             "semantic_original_vs_rewritten": semantic["values"][index]}
            for name in before for index, record in enumerate(records)]
    with (directory / "paired_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    count = sum(len(row["edits"]) for row in records)
    result = {"records": metrics, "num_pairs": len(pairs), "num_changed": sum(changed), "num_edits": count,
              "actual_fraction_of_eligible_words": count / sum(row["eligible_content_words"] for row in records),
              "mean_semantic_original_vs_rewritten": float(np.mean(semantic["values"])),
              "mean_semantic_changed_only": float(np.asarray(semantic["values"])[changed].mean()),
              "num_rejected_proposals": sum(len(row["rejections"]) for row in records),
              "variants_sha256": frozen, "completed_at": datetime.now(timezone.utc).isoformat()}
    write_record(result_path, result)
    return result


def run(path):
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
    from scripts.research_workflow import source_provenance
    from scripts.report_research import update_report

    path = Path(path)
    manifest = json.loads(path.read_text())
    job = os.environ["SLURM_JOB_ID"]
    if job != manifest["job_id"]:
        raise ValueError("Synonym evaluation does not match its Slurm job")
    with (path.parent / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = path.parent / "receipt.json"
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        if previous.get("status") == "completed":
            return
        receipt = {"status": "running", "attempt": previous.get("attempt", 0) + 1,
                   "slurm_job_id": job, "started_at": datetime.now(timezone.utc).isoformat()}
        write_record(receipt_path, receipt)
        try:
            with StopRequest() as stop:
                receipt.update(source_provenance())
                write_record(receipt_path, receipt)
                experiment(manifest, path.parent, stop)
            receipt["status"] = "completed"
        except CheckpointPause as error:
            receipt.update(status="paused", error=str(error))
        except BaseException as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_record(receipt_path, receipt)
            write_record(path.parent / f"attempt-{receipt['attempt']}.json", receipt)
            update_report()
        if receipt["status"] == "paused":
            if receipt["attempt"] >= manifest["max_restarts"]:
                raise RuntimeError("Synonym evaluation reached its restart limit")
            subprocess.run(["scontrol", "requeue", job], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("submit")
    for mode in ("release", "run"):
        modes.add_parser(mode).add_argument("manifest", type=Path)
    arguments = parser.parse_args()
    if arguments.mode == "submit":
        submit()
    elif arguments.mode == "release":
        release(arguments.manifest)
    else:
        run(arguments.manifest)
