"""Checkpointed, small A/B timing and reward-equivalence test on one H100."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from scripts.research_chain import ROOT, OUTPUT, write_record

DEFENDERS = {"gemma": "decoder_classifier", "diveye": "diveye"}


class ScalarSemantic:
    def __init__(self, encoder):
        self.encoder = encoder

    def __call__(self, source, generated):
        return self.encoder(source, generated)


def submit():
    directory = OUTPUT / "defender_comparison/optimization_profile" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / "manifest.json"
    manifest = {"steps_per_case": 6, "warmup_steps": 2, "defenders": DEFENDERS,
                "logging_subset_size": 32, "logging_interval": 50,
                "held_races": ["1271274", "1271276", "1271277"], "max_restarts": 20}
    write_record(path, manifest)
    command = ["sbatch", "--parsable", "--hold", "--qos=qos_gpu_h100-dev", "--time=02:00:00",
               "--signal=USR1@600", "--requeue", "--cpus-per-task=32", "--job-name=rl-optimization-profile",
               f"--output={directory}/slurm-%j.log", str(ROOT / "scripts/run_research_slurm.sh"),
               "--profile-rl-optimizations", str(path)]
    job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";", 1)[0]
    if not job.isdigit():
        raise ValueError(f"Unexpected Slurm response: {job!r}")
    manifest.update(job_id=job, command=command, held=True)
    write_record(path, manifest)
    print(path, job, flush=True)


def reward_equivalence(plan, name, directory):
    import numpy as np
    from ai_content_detector.rl_evasion.defender_comparison import workflow
    from ai_content_detector.rl_evasion.text_evasion.rewards import SemanticSimilarityReward

    result_path = directory / "reward_equivalence.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    rows = json.loads((workflow.OUTPUT / "base_pairs.json").read_text())["evaluation"][:16]
    sources = [row["reference"] for row in rows] + [rows[0]["reference"]]
    texts = [row["ai"] for row in rows] + ["too short"]
    plan = deepcopy(plan)
    if name == "diveye":
        plan["specifications"]["diveye"]["feature_cache"] = str(directory / "reference_features")
    pool = workflow.group_pool(plan, name, directory / "defenders", names=[DEFENDERS[name]])
    reward = workflow.BatchReward(pool)
    try:
        pool.keep_on_device = False
        reward.semantic = ScalarSemantic(SemanticSimilarityReward(device="cpu"))
        before = reward.batch(sources, texts)
        if name == "diveye":
            pool.specifications["diveye"]["feature_cache"] = str(directory / "optimized_features")
        pool.keep_on_device = True
        reward.semantic = SemanticSimilarityReward(device="cpu")
        after = reward.batch(sources, texts)
        differences = {}
        for key in ("total", "evasion", "semantic", "quality"):
            differences[key] = float(np.max(np.abs(np.asarray([row[key] for row in before]) -
                                                   np.asarray([row[key] for row in after]))))
            if differences[key] > 1e-6:
                raise ValueError(f"Reward changed beyond numerical tolerance: {key}: {differences[key]}")
        if any(a["rejected"] != b["rejected"] or a["per_detector"] != b["per_detector"]
               for a, b in zip(before, after)):
            raise ValueError("Detector scores or short-output rejection changed")
        value = {"passed": True, "absolute_tolerance": 1e-6, "max_absolute_differences": differences,
                 "num_examples": len(texts), "baseline": before, "optimized": after}
        write_record(result_path, value)
        print(name, "reward equivalence", differences, flush=True)
        return value
    finally:
        pool.clear()


def profile_case(plan, name, variant, manifest, directory, original_kernels, stop):
    import torch
    from ai_content_detector.rl_evasion.defender_comparison import workflow
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause

    result_path = directory / "results.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    for module, functions in original_kernels:
        for function, implementation in functions.items():
            setattr(module, function, inspect.unwrap(implementation) if variant == "reference" else implementation)
    plan = deepcopy(plan)
    if name == "diveye":
        plan["specifications"]["diveye"]["feature_cache"] = str(directory / "diveye_features")
    attacker, pool = workflow.attacker_setup(plan, name, directory / "attacker", names=[DEFENDERS[name]])
    try:
        attacker.config.num_train_epochs = 1
        attacker.config.logging_steps = 1
        if variant == "reference":
            pool.keep_on_device = False
            attacker.reward_fn.semantic = ScalarSemantic(attacker.reward_fn.semantic)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        attacker.train(max_steps=manifest["steps_per_case"])
        if stop.requested:
            raise CheckpointPause("Optimization profile retained native optimizer progress")
        history = attacker.training_history
        for row in history:
            for key in ("loss", "grad_norm", "reward", "step_time"):
                if key in row and not math.isfinite(float(row[key])):
                    raise ValueError(f"Nonfinite training metric: {name}/{variant}/{key}")
        if any(not torch.isfinite(parameter).all() for parameter in attacker.model.parameters() if parameter.requires_grad):
            raise ValueError("Nonfinite attacker adapter weights")
        selected = [row["step_time"] for row in history
                    if row.get("step", 0) > manifest["warmup_steps"] and "step_time" in row]
        if len(selected) < manifest["steps_per_case"] - manifest["warmup_steps"]:
            raise ValueError("Missing timed optimizer updates")
        diagnostic = directory / "attacker/diagnostics" / f"step_{manifest['steps_per_case']:06d}/metrics.json"
        value = {"defender": name, "variant": variant, "steps": manifest["steps_per_case"],
                 "warmup_steps": manifest["warmup_steps"], "timed_step_seconds": selected,
                 "mean_step_seconds": statistics.mean(selected), "median_step_seconds": statistics.median(selected),
                 "projected_400_update_minutes": statistics.mean(selected) * 400 / 60,
                 "train_call_seconds": time.perf_counter() - started,
                 "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                 "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
                 "diagnostic": json.loads(diagnostic.read_text())}
        write_record(directory / "training_history.json", history)
        write_record(result_path, value)
        print(name, variant, value["mean_step_seconds"], "seconds/update", flush=True)
        return value
    finally:
        pool.clear()
        del attacker
        gc.collect()
        torch.cuda.empty_cache()


def run(path):
    import torch
    import causal_conv1d
    import fla
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_next import modeling_qwen3_next
    from ai_content_detector.rl_evasion.checkpointing import StopRequest, CheckpointPause
    from ai_content_detector.rl_evasion.defender_comparison import workflow
    from scripts.research_workflow import source_provenance

    path = Path(path)
    manifest = json.loads(path.read_text())
    if os.environ["SLURM_JOB_ID"] != manifest["job_id"]:
        raise ValueError("Optimization profile Slurm identity changed")
    kernels = ["causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule",
               "torch_recurrent_gated_delta_rule"]
    originals = [(module, {name: getattr(module, name) for name in kernels})
                 for module in (modeling_qwen3_5, modeling_qwen3_next)]
    dispatch = {}
    for module, functions in originals:
        for name, function in functions.items():
            selected = None
            while hasattr(function, "__wrapped__"):
                closure = inspect.getclosurevars(function).nonlocals
                if closure.get("is_new_implementation"):
                    selected = closure["implementation"].__module__
                    break
                function = function.__wrapped__
            if selected is None:
                raise ValueError(f"Optimized kernel is unavailable: {module.__name__}.{name}")
            dispatch[module.__name__ + "." + name] = selected
    receipt_path = path.parent / "receipt.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    receipt = {"status": "running", "attempt": previous.get("attempt", 0) + 1,
               "optimized_dispatch": dispatch,
               "versions": {"torch": torch.__version__, "causal_conv1d": causal_conv1d.__version__, "fla": fla.__version__},
               "started_at": datetime.now(timezone.utc).isoformat()}
    write_record(receipt_path, receipt)
    try:
        receipt.update(source_provenance())
        write_record(receipt_path, receipt)
        with StopRequest() as stop:
            plan = workflow.load_plan()
            results = []
            for name in DEFENDERS:
                reward_equivalence(plan, name, path.parent / name)
                for variant in ("reference", "optimized"):
                    results.append(profile_case(plan, name, variant, manifest,
                                                path.parent / name / variant, originals, stop))
            write_record(path.parent / "results.json", {"cases": results, "logging_only": True,
                         "note": "Six-step performance test; not evidence of learned evasion or a production checkpoint."})
        receipt["status"] = "completed"
        subprocess.run(["scontrol", "release", *manifest["held_races"]], check=True)
    except CheckpointPause as error:
        receipt.update(status="paused", error=str(error))
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for module, functions in originals:
            for name, implementation in functions.items():
                setattr(module, name, implementation)
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_record(receipt_path, receipt)
    if receipt["status"] == "paused":
        if receipt["attempt"] >= manifest["max_restarts"]:
            raise RuntimeError("Optimization profile exceeded its restart budget")
        subprocess.run(["scontrol", "requeue", manifest["job_id"]], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["submit", "run"])
    parser.add_argument("manifest", nargs="?")
    arguments = parser.parse_args()
    submit() if arguments.mode == "submit" else run(arguments.manifest)
