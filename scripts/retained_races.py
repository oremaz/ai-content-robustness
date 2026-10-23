"""Prepare three defender-only updates from the retained long-run attackers."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts import architecture_races
from scripts.research_chain import OUTPUT, ROOT, write_record

SOURCES = {"gemma": ("decoder", 1213), "modernbert": ("encoder", 1487), "ensemble": ("ensemble", 502)}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cached_measurement(pairs, artifact, names, thresholds, semantic=None):
    import numpy as np
    from ai_content_detector.rl_evasion.defender_comparison.protocol import classification_metrics

    eligible = [row for row in pairs if row["eligible"]]
    if artifact["eligible_ids"] != [row["sample_id"] for row in eligible] or not eligible:
        raise ValueError("Cached detector scores do not match the saved evaluation pairs")
    scores = {name: np.asarray(artifact["scores"][name], dtype=float) for name in names}
    n = len(eligible)
    if any(values.shape != (2 * n,) or not np.isfinite(values).all()
           or (values < 0).any() or (values > 1).any() for values in scores.values()):
        raise ValueError("Invalid cached detector scores")

    def record(name, values):
        row = {"detector": name, **classification_metrics(values[:n], values[n:]),
               "num_generated": len(pairs), "num_ineligible_pairs": len(pairs) - n}
        row["conditional_attack_success_rate"] = row["attack_success_rate"]
        row["attack_success_rate"] = float(sum(values[:n] < 0.5) / len(pairs))
        if name in thresholds:
            operating = classification_metrics(values[:n], values[n:], thresholds[name])
            row.update(tpr_at_validation_1pct_fpr=operating["ai_detection_rate"],
                       test_fpr_at_validation_threshold=operating["human_false_positive_rate"],
                       validation_threshold=thresholds[name])
        return row

    records = [record(name, values) for name, values in scores.items()]
    ensemble = record("ensemble_mean", np.mean(list(scores.values()), axis=0))
    ensemble["valid_evasion_fraction_of_all_outputs"] = ensemble["attack_success_rate"]
    return {"records": records + [ensemble], "mean_semantic_similarity": semantic,
            "mean_ai_words": float(np.mean([len(row["ai"].split()) for row in pairs])),
            "mean_human_words": float(np.mean([len(row["human"].split()) for row in pairs]))}


def prepare():
    from ai_content_detector.rl_evasion.checkpointing import load_checkpoint
    from ai_content_detector.rl_evasion.config import TextEvasionConfig
    from ai_content_detector.rl_evasion.defender_comparison import workflow
    from ai_content_detector.rl_evasion.defender_comparison.protocol import digest

    identity, plan = architecture_races.inputs()
    root = workflow.OUTPUT
    all_initial = workflow.checkpoint_identity(workflow.warm_checkpoints())
    if json.loads((root / "warmup_identity.json").read_text()) != all_initial:
        raise ValueError("The warmed defenders differ from their completed evaluation")
    base = json.loads((root / "base_pairs.json").read_text())["evaluation"]
    warm_scores = json.loads((root / "warmup/measurements.json").read_text())
    thresholds = json.loads((root / "validation_thresholds.json").read_text())
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = OUTPUT / "defender_comparison/races_100" / identifier / "manifest.json"
    defenders, bundles = {}, {}
    defaults = TextEvasionConfig()
    for name, (group, steps) in SOURCES.items():
        evaluation = root / "checkpoint_evaluations" / group / f"step-{steps}"
        result = json.loads((evaluation / "results.json").read_text())
        metadata = result["checkpoint"]
        checkpoint = root / "rl" / group / f"checkpoint-{steps}"
        if (metadata["checkpoint"] != str(checkpoint) or metadata["global_step"] != steps
                or not (checkpoint / "restart_complete.pt").is_file()):
            raise ValueError("Retained checkpoint identity differs from its completed evaluation")
        for filename, expected in metadata["checkpoint_sha256"].items():
            if sha256(checkpoint / filename) != expected:
                raise ValueError("Retained checkpoint changed after evaluation")
        for filename, expected in metadata["evaluation_inputs_sha256"].items():
            if sha256(root / filename) != expected:
                raise ValueError("Evaluation partition changed after the retained run")
        config = metadata["config"]
        for key in ("generator_model", "max_seq_length", "max_new_tokens", "temperature", "top_p", "seed"):
            if config[key] != getattr(defaults, key):
                raise ValueError(f"Retained generation configuration differs: {key}")
        names = config["detector_names"]
        if digest({key: all_initial[key] for key in names}) != config["detector_checkpoint_hash"]:
            raise ValueError("Retained attack used different warmed defenders")
        pairs = load_checkpoint(evaluation / "semantic_diagnostics.pt")["config"]["pairs"]
        if len(pairs) != len(base) or any(
            any(row[key] != original[key] for key in ("prompt", "reference", "human", "sample_id"))
            for row, original in zip(pairs, base)
        ):
            raise ValueError("Retained texts do not match the reserved evaluation partition")
        directory = path.parent / "initial_evaluations" / name
        files = {"pairs": pairs,
                 "before": cached_measurement(base, warm_scores, names, thresholds),
                 "after_attack": cached_measurement(pairs,
                     json.loads((evaluation / "scores/measurements.json").read_text()), names,
                     thresholds, result["mean_semantic_similarity"])}
        locations = {}
        for phase, value in files.items():
            target = directory / f"{phase}.json"
            write_record(target, value)
            locations[phase] = str(target)
        artifacts = [checkpoint / filename for filename in metadata["checkpoint_sha256"]]
        artifacts += [Path(target) for target in locations.values()]
        artifacts += [evaluation / "results.json"]
        bundles[name] = {"checkpoint": str(checkpoint), "updates": steps, **locations,
                         "sha256": {str(file): sha256(file) for file in artifacts}}
        defenders[name] = names
    protocol = {"rounds": 1, "attacker_steps_per_round": 0, "initialization": "retained_attackers",
                "defenders": defenders, "evaluation_prompts": 500, "defender_training_prompts": 1000,
                "seed": 42, "execution": {"generation_batch_size": 8, "batch_semantic_evaluation": True,
                                          "reuse_before_evaluation": True}}
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                "inputs": identity, "baseline_plan": plan, "all_initial_defenders": all_initial,
                "warm_starts": bundles, "qos": "qos_gpu_h100-dev", "time_limit": "02:00:00",
                "gpu_limit": 3, "max_restarts": 100, "cpus_by_branch": {"ensemble": 48},
                "metrics_file": str(path.parent / "metrics.csv"), "held": True, "jobs": {}}
    write_record(path, manifest)
    architecture_races.write_metrics(Path(manifest["metrics_file"]), [])
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("prepare")
    submission = commands.add_parser("submit")
    submission.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(prepare())
    else:
        manifest = json.loads(args.manifest.read_text())
        if manifest["jobs"]:
            raise ValueError("This defender-only run has already been submitted")
        architecture_races.submit_jobs(args.manifest, manifest)
