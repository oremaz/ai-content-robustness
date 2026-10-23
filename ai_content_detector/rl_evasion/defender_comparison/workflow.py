"""Execute the agreed seven-defender experiment in resumable Slurm stages."""

from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .defenders import DefenderPool, TorchDefender
from .plan import GROUPS, specifications
from .protocol import (classification_metrics, digest, evaluate_pool, generation, matched_pairs,
                       prepare_pairs, score_checkpointed, validation_threshold, write_json)
from ..checkpointing import (CheckpointPause, StopRequest, atomic_save, load_checkpoint,
                             model_state, restore_model, restore_rng, rng_state)
from ..config import TextEvasionConfig
from ..text_evasion.grpo_trainer import GRPOTextEvasionTrainer
from ..text_evasion.rewards import SemanticSimilarityReward

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/research_workflow/defender_comparison"


def load_plan():
    from scripts.defender_selection import save_notebook_selection
    for stage in ("text_notebook",):
        receipt = json.loads((OUTPUT.parent / f"{stage}.json").read_text())
        if receipt.get("status") != "completed":
            raise RuntimeError(f"{stage} must finish before the defender comparison")
    selection_path = OUTPUT / "notebook_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        measurements = (OUTPUT.parent / "text_notebook_measurements.json").read_bytes()
        if selection["measurements_sha256"] != hashlib.sha256(measurements).hexdigest():
            raise ValueError("Notebook selection does not match the completed measurements")
    else:
        selection = save_notebook_selection(OUTPUT.parent / "text_notebook_measurements.json",
                                           ROOT / "ai_content_detector/notebooks/bench-aitextdetect.ipynb",
                                           selection_path)
    plan = {"specifications": specifications(selection, OUTPUT), "groups": GROUPS,
            "generator": TextEvasionConfig().generator_model, "seed": 42,
            "warmup_examples": 2000, "defender_epochs": 3, "defender_learning_rate": 2e-5,
            "warmup_validation": "paired_80_20", "standalone_rl_epochs": 3,
            "race_rounds": 10, "attacker_steps_per_round": 500,
            "race_initialization": "base_Qwen_or_exact_compatible_step_500",
            "selection": selection}
    plan_path = OUTPUT / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Defender comparison plan differs from the saved experiment")
    else:
        write_json(plan_path, plan)
    return plan


def warm_checkpoints():
    return json.loads((OUTPUT / "warmup_checkpoints.json").read_text())


class BatchReward:
    def __init__(self, pool):
        self.pool = pool
        self.semantic = SemanticSimilarityReward(device="cpu")
        if isinstance(pool, DefenderPool) and len(pool.specifications) == 1:
            pool.keep_on_device = True

    def batch(self, sources, generated):
        if len(sources) != len(generated):
            raise ValueError("Reward sources and completions must be aligned")
        valid = [index for index, text in enumerate(generated) if len(text.split()) >= 20]
        scores = self.pool.predict_batch([generated[index] for index in valid]) if valid else {}
        positions = {index: offset for offset, index in enumerate(valid)}
        if hasattr(self.semantic, "batch"):
            semantic_values = self.semantic.batch([sources[index] for index in valid],
                                                  [generated[index] for index in valid])
        else:
            semantic_values = [self.semantic(sources[index], generated[index]) for index in valid]
        result = []
        for index, (source, text) in enumerate(zip(sources, generated)):
            if index not in positions:
                result.append({"total": 0.0, "evasion": 0.0, "semantic": 0.0, "quality": 0.0,
                               "rejected": True, "per_detector": {name: 0.0 for name in self.pool.specifications}})
                continue
            evasion = {name: 1.0 - values[positions[index]] for name, values in scores.items()}
            mean_evasion = float(np.mean(list(evasion.values())))
            semantic = semantic_values[positions[index]]
            ratio = len(text.split()) / max(len(source.split()), 1)
            quality = max(0.0, 1.0 - (1.0 - ratio) ** 2)
            result.append({"total": 0.6 * mean_evasion + 0.3 * semantic + 0.1 * quality,
                           "evasion": mean_evasion, "semantic": semantic, "quality": quality,
                           "rejected": False, "per_detector": evasion})
        return result

    def __call__(self, source, generated):
        return self.batch([source], [generated])[0]


class ComparisonAttacker(GRPOTextEvasionTrainer):
    def __init__(self, config, pool):
        os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".venv/cache/triton"))
        super().__init__(config)
        self.pool = pool

    def _setup_rewards(self):
        self.reward_fn = BatchReward(self.pool)

    def log_training_diagnostic(self, step):
        from .diagnostics import log_training_diagnostic
        return log_training_diagnostic(self, step)


def group_pool(plan, group, root, checkpoints=None, names=None):
    names = names if names is not None else GROUPS[group] or list(plan["specifications"])
    return DefenderPool({name: plan["specifications"][name] for name in names}, root,
                        checkpoints=checkpoints or {name: warm_checkpoints()[name] for name in names})


def checkpoint_identity(checkpoints):
    identity = {}
    for name, directory in checkpoints.items():
        path = Path(directory)
        files = [path / "completed.pt"]
        if (path / "classifier.pkl").exists():
            files.append(path / "classifier.pkl")
        identity[name] = {"path": str(path), "files": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                                                      for file in files}}
    return identity


def baseline(plan, stop=None):
    pairs = prepare_pairs(OUTPUT, stop=stop)
    pool = DefenderPool(plan["specifications"], OUTPUT / "initial_defenders", retain_on_cpu=False)
    try:
        return {"records": evaluate_pool(pool, pairs["evaluation"], OUTPUT / "baseline"),
                "interpretation": "Transfer to unchanged Qwen before Qwen-specific defender adaptation."}
    finally:
        pool.clear()


def warmup(plan, stop=None):
    from sklearn.model_selection import train_test_split
    pairs = json.loads((OUTPUT / "base_pairs.json").read_text())
    training = [row for row in pairs["warmup"] if row["eligible"]]
    if len(training) < 5:
        raise ValueError("Insufficient eligible base-Qwen examples for defender adaptation")
    mapping_path = OUTPUT / "warmup_checkpoints.json"
    checkpoints = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    pool = DefenderPool(plan["specifications"], OUTPUT / "adapted_defenders", retain_on_cpu=False)
    history_path = OUTPUT / "warmup_training.json"
    histories = json.loads(history_path.read_text()) if history_path.exists() else {}
    try:
        for name in pool.specifications:
            if name in checkpoints:
                continue
            try:
                defender = pool.get(name)
                if stop is not None and stop.requested:
                    raise CheckpointPause("Defender setup paused before warm-up")
                histories[name] = defender.retrain([row["ai"] for row in training],
                                                    [row["human"] for row in training], warmup=True)
                checkpoints[name] = str(OUTPUT / "adapted_defenders" / name / "warmup")
                write_json(mapping_path, checkpoints)
                write_json(OUTPUT / "warmup_training.json", histories)
            finally:
                pool.release(name)
        pool.clear()
        pool = DefenderPool(plan["specifications"], OUTPUT / "adapted_defenders", checkpoints=checkpoints,
                            retain_on_cpu=False)
        _, validation = train_test_split(np.arange(len(training)), test_size=0.2, random_state=42)
        validation_scores = score_checkpointed(pool, [training[index]["ai"] for index in validation] +
                                               [training[index]["human"] for index in validation], OUTPUT / "validation")
        write_json(OUTPUT / "validation_scores.json", validation_scores)
        thresholds = {name: validation_threshold(values[len(validation):])
                      for name, values in validation_scores.items()}
        write_json(OUTPUT / "validation_thresholds.json", thresholds)
        write_json(OUTPUT / "warmup_identity.json", checkpoint_identity(checkpoints))
        return {"records": evaluate_pool(pool, pairs["evaluation"], OUTPUT / "warmup", thresholds),
                "interpretation": "Same base-Qwen evaluation examples after supervised defender warm-up."}
    finally:
        pool.clear()


def attacker_setup(plan, group, directory, names=None):
    pool = group_pool(plan, group, directory / "defenders", names=names)
    configuration = TextEvasionConfig(output_dir=str(directory), detector_names=list(pool.specifications),
                                      detector_checkpoint_hash=digest(checkpoint_identity(pool.checkpoints)))
    attacker = ComparisonAttacker(configuration, pool)
    attacker.setup()
    manifest = json.loads((OUTPUT / "prompt_manifest.json").read_text())
    if attacker.prompts != manifest["train"] or attacker.eval_prompts != manifest["evaluation"]:
        raise ValueError("RL prompt partition differs from the recorded baseline")
    return attacker, pool


def evaluate_attacker(attacker, pool, directory, context, generation_directory=None, evaluation_pairs=None):
    base = json.loads((OUTPUT / "base_pairs.json").read_text())["evaluation"]
    if evaluation_pairs is None:
        outputs = generation(attacker, [row["prompt"] for row in base], generation_directory or directory, context, seed=42)
        pairs = [{**row, "ai": text, "eligible": row["eligible"] and len(text.split()) >= 20}
                 for row, text in zip(base, outputs)]
    else:
        pairs = evaluation_pairs
        if len(pairs) != len(base) or any(
            any(row.get(key) != original.get(key) for key in ("prompt", "reference", "human", "sample_id"))
            or row["eligible"] != (original["eligible"] and len(row["ai"].split()) >= 20)
            for row, original in zip(pairs, base)
        ):
            raise ValueError("Saved evaluation pairs differ from the reserved evaluation partition")
    thresholds = json.loads((OUTPUT / "validation_thresholds.json").read_text())
    records = evaluate_pool(pool, pairs, directory / "scores", thresholds)
    valid = [row for row in pairs if row["eligible"]]
    scores = score_checkpointed(pool, [row["ai"] for row in valid] + [row["human"] for row in valid],
                                directory / "scores")
    mean = np.mean(list(scores.values()), axis=0)
    ensemble = classification_metrics(mean[:len(valid)], mean[len(valid):])
    ensemble["conditional_attack_success_rate"] = ensemble["attack_success_rate"]
    ensemble["attack_success_rate"] = float(sum(mean[:len(valid)] < 0.5) / len(pairs))
    ensemble.update({"detector": "ensemble_mean", "num_generated": len(pairs),
                     "num_ineligible_pairs": len(pairs) - len(valid),
                     "valid_evasion_fraction_of_all_outputs": float(sum(mean[:len(valid)] < 0.5) / len(pairs))})
    records.append(ensemble)
    semantic = SemanticSimilarityReward(device="cpu")
    if getattr(attacker, "batch_semantic_evaluation", False):
        from .evaluation import semantic_values
        values = semantic_values(attacker.reward_fn.semantic, pairs, directory.parent.parent / "semantic_cache")
        return {"records": records, "mean_semantic_similarity": float(np.mean(values)) if values else None,
                "mean_ai_words": float(np.mean([len(row["ai"].split()) for row in pairs])),
                "mean_human_words": float(np.mean([len(row["human"].split()) for row in pairs]))}
    diagnostics_path = directory / "semantic_diagnostics.pt"
    signature = {"pairs": pairs}
    diagnostics = load_checkpoint(diagnostics_path, signature) or {"config": signature, "values": []}
    with StopRequest() as stop:
        for row in pairs[len(diagnostics["values"]):]:
            if stop.requested:
                raise CheckpointPause("Semantic evaluation checkpoint retained")
            diagnostics["values"].append(semantic(row["reference"], row["ai"]) if row["ai"].strip() else None)
            atomic_save(diagnostics, diagnostics_path)
    semantic_values = [value for value in diagnostics["values"] if value is not None]
    return {"records": records, "mean_semantic_similarity": float(np.mean(semantic_values)) if semantic_values else None,
            "mean_ai_words": float(np.mean([len(row["ai"].split()) for row in pairs])),
            "mean_human_words": float(np.mean([len(row["human"].split()) for row in pairs]))}


def standalone(plan, group, stop=None):
    directory = OUTPUT / "rl" / group
    attacker, pool = attacker_setup(plan, group, directory)
    try:
        if stop is not None and stop.requested:
            raise CheckpointPause("Attacker setup paused before training")
        attacker.train()
        write_json(directory / "training_history.json", attacker.training_history)
        return evaluate_attacker(attacker, pool, directory / "evaluation", f"rl/{group}/final")
    finally:
        pool.clear()


def milestone_reuse_decision(directory, config, checkpoints):
    milestone = Path(directory) / "milestones/step_500"
    state_path = milestone / "trainer_state.json"
    if not state_path.exists():
        return {"reuse": False, "reason": "No preserved step-500 checkpoint", "checkpoint": str(milestone)}
    state = json.loads(state_path.read_text())
    context = load_checkpoint(milestone / "research_restart_context.pt")
    original = asdict(TextEvasionConfig(**context["config"]))
    current = asdict(config)
    differences = [key for key in current if key not in {"output_dir", "num_train_epochs"}
                   and current[key] != original.get(key)]
    if state["max_steps"] != 500:
        differences.append("learning_rate_schedule_horizon")
    if state["global_step"] != 500:
        differences.append("optimizer_step")
    # Matching horizon and all training inputs are required, not only LoRA weights.
    return {"reuse": not differences, "reason": "Matching first-round protocol" if not differences else
            "Incompatible: " + ", ".join(differences), "checkpoint": str(milestone),
            "source_horizon": state["max_steps"], "race_round_horizon": 500,
            "defender_identity": checkpoint_identity(checkpoints)}


def race(plan, group, stop=None, *, directory=None, defender_names=None, rounds=10,
         attacker_steps=500, reuse_standalone=True, on_round=None, on_measurement=None,
         pause_after_round=None, pause_after_attacker_round=None, warm_start=None):
    if rounds < 1 or attacker_steps < 0 or (attacker_steps == 0 and not (warm_start and rounds == 1)):
        raise ValueError("Zero attacker updates require a single defender-only warm-start round")
    directory = Path(directory) if directory is not None else OUTPUT / "race" / group
    if defender_names is None:
        attacker, pool = attacker_setup(plan, group, directory / "attacker")
    else:
        attacker, pool = attacker_setup(plan, group, directory / "attacker", names=defender_names)
    if stop is not None and stop.requested:
        raise CheckpointPause("Race setup paused before training")
    attacker.config.num_train_epochs = 1
    execution = plan.get("execution", {})
    training = execution.get("attacker_training", {})
    if set(training) - {"learning_rate", "lr_scheduler_type", "warmup_steps", "lr_scheduler_kwargs",
                        "grpo_beta", "grpo_num_generations", "per_device_train_batch_size",
                        "gradient_accumulation_steps"}:
        raise ValueError("Unsupported attacker training override")
    for name, value in training.items():
        setattr(attacker.config, name, value)
    if attacker_steps and not 0 <= attacker.config.warmup_steps < attacker_steps:
        raise ValueError("Attacker warm-up must be shorter than its update budget")
    attacker.generation_batch_size = execution.get("generation_batch_size", 1)
    attacker.batch_semantic_evaluation = execution.get("batch_semantic_evaluation", False)
    attacker.diagnostic_samples = execution.get("diagnostic_samples", 32)
    attacker.microbatch_candidates = execution.get("microbatch_candidates", [])
    if pause_after_round is not None and not 1 <= pause_after_round <= rounds:
        raise ValueError("Pilot pause must be within the planned race rounds")
    if pause_after_attacker_round is not None and not 1 <= pause_after_attacker_round <= rounds:
        raise ValueError("Attacker pause must be within the planned race rounds")
    restart_path = directory / "race_restart.pt"
    configuration = {"plan": plan, "group": group, "initial_defenders": pool.checkpoints,
                     "rounds": rounds, "attacker_steps": attacker_steps,
                     "reuse_standalone": reuse_standalone}
    if warm_start is not None:
        from . import warm_start as retained
        retained.verify(warm_start)
        configuration["warm_start"] = warm_start
    state = load_checkpoint(restart_path, configuration)
    if state is None:
        decision = (milestone_reuse_decision(OUTPUT / "rl" / group, attacker.config, pool.checkpoints)
                    if reuse_standalone and attacker_steps == 500 else
                    {"reuse": False, "reason": "Start from base Qwen for the recorded race protocol"})
        if warm_start is not None:
            retained.load_policy(attacker, warm_start)
            decision = {"reuse": True, "reason": "Retained attacker; begin with its first defender update",
                        "checkpoint": warm_start["checkpoint"], "updates": warm_start["updates"]}
        write_json(directory / "reuse_decision.json", decision)
        state = {"config": configuration, "round": 0, "phase": "before", "history": [], "metrics": {},
                 "checkpoints": dict(pool.checkpoints), "reuse": decision, "attacker_updates": 0}
    else:
        restore_model(attacker.model, state["attacker"])
        restore_rng(state["rng"])
        pool.checkpoints = dict(state["checkpoints"])
    base = json.loads((OUTPUT / "base_pairs.json").read_text())

    def save():
        state["attacker"] = model_state(attacker.model)
        state["rng"] = rng_state()
        atomic_save(state, restart_path)

    save()
    try:
        with StopRequest() as stop:
            while state["round"] < rounds:
                if pause_after_round is not None and state["round"] >= pause_after_round:
                    break
                if state["round"] + 1 == pause_after_attacker_round and state["phase"] == "generate_defender":
                    break
                if stop.requested:
                    raise CheckpointPause("Detector-comparison race state retained")
                index, phase = state["round"], state["phase"]
                round_dir = directory / f"round_{index + 1}"
                print(f"{group}: round {index + 1}/{rounds}; phase {phase}", flush=True)
                if phase in {"before", "after_attack"}:
                    if index == 0 and warm_start is not None:
                        measurement = retained.read(warm_start, phase)
                        write_json(round_dir / phase / "reuse.json", {"warm_start": warm_start,
                                   "phase": phase, "source_updates": 0 if phase == "before" else warm_start["updates"]})
                    elif phase == "before" and index > 0 and execution.get("reuse_before_evaluation"):
                        from .evaluation import evaluation_identity
                        identity = evaluation_identity(attacker, pool, base["evaluation"])
                        if identity != state.get("after_defense_identity"):
                            raise ValueError("Cannot reuse evaluation after attacker, defender or inputs changed")
                        measurement = state["history"][-1]["after_defense"]
                        write_json(round_dir / phase / "reuse.json", {"source_round": index,
                                   "source_phase": "after_defense", "identity": identity})
                    else:
                        measurement = evaluate_attacker(attacker, pool, round_dir / phase,
                                                       f"race/{group}/{index}/{phase}")
                    measurement = {**measurement, "attacker_updates": state.get("attacker_updates", index * attacker_steps)}
                    write_json(round_dir / phase / "evaluation.json", measurement)
                    if on_measurement is not None:
                        on_measurement(index + 1, phase, measurement)
                    state["metrics"][phase] = measurement
                    if phase == "before" and index == 0 and warm_start is not None:
                        state.update(phase="after_attack", attacker_updates=warm_start["updates"])
                    else:
                        state["phase"] = "attack" if phase == "before" else "generate_defender"
                elif phase == "attack":
                    if index == 0 and state["reuse"]["reuse"]:
                        from peft import set_peft_model_state_dict, load_peft_weights
                        weights = load_peft_weights(state["reuse"]["checkpoint"], device="cpu")
                        set_peft_model_state_dict(attacker.model, weights)
                    else:
                        attacker.config.output_dir = str(round_dir / "attacker")
                        attacker.config.detector_checkpoint_hash = digest(checkpoint_identity(pool.checkpoints))
                        attacker.train(max_steps=attacker_steps)
                        write_json(round_dir / "training_history.json", attacker.training_history)
                    updates = max((row.get("step", 0) for row in attacker.training_history), default=0)
                    state["attacker_updates"] = state.get("attacker_updates", index * attacker_steps) + int(updates or attacker_steps)
                    state["phase"] = "after_attack"
                elif phase == "generate_defender":
                    initial = base["warmup"]
                    outputs = generation(attacker, [row["prompt"] for row in initial], round_dir,
                                         f"defender_training/{index}", seed=42 + index + 1)
                    state["training"] = matched_pairs([row["prompt"] for row in initial],
                                                       [row["reference"] for row in initial], outputs, attacker.tokenizer)
                    state["phase"] = "defend"
                elif phase == "defend":
                    training = [row for row in state["training"] if row["eligible"]]
                    for name in pool.specifications:
                        target = str(round_dir / "defenders" / name / f"round_{index + 1}")
                        if state["checkpoints"][name] == target:
                            continue
                        defender = pool.get(name)
                        defender.output_dir = str(round_dir / "defenders" / name) if isinstance(defender, TorchDefender) \
                            else round_dir / "defenders" / name
                        defender._round = index
                        defender.retrain([row["ai"] for row in training], [row["human"] for row in training])
                        pool.release(name)
                        state["checkpoints"][name] = target
                        pool.checkpoints = dict(state["checkpoints"])
                        save()
                    pool.clear()
                    state["phase"] = "after_defense"
                elif phase == "after_defense":
                    # Reuse the exact post-attack texts; only defender weights change.
                    options = {"evaluation_pairs": retained.read(warm_start, "pairs")} \
                        if index == 0 and warm_start is not None else {}
                    measurement = evaluate_attacker(
                        attacker, pool, round_dir / "after_defense", f"race/{group}/{index}/after_attack",
                        generation_directory=round_dir / "after_attack", **options)
                    measurement["attacker_updates"] = state.get("attacker_updates", (index + 1) * attacker_steps)
                    write_json(round_dir / phase / "evaluation.json", measurement)
                    if on_measurement is not None:
                        on_measurement(index + 1, phase, measurement)
                    state["metrics"][phase] = measurement
                    if execution.get("reuse_before_evaluation"):
                        from .evaluation import evaluation_identity
                        state["after_defense_identity"] = evaluation_identity(attacker, pool, base["evaluation"])
                    state["history"].append({"round": index + 1, **state["metrics"]})
                    write_json(directory / "history.json", state["history"])
                    state.update(round=index + 1, phase="before", metrics={})
                    state.pop("training", None)
                else:
                    raise ValueError(f"Unknown comparison race phase {phase}")
                save()
                if phase == "after_defense" and on_round is not None:
                    on_round()
    finally:
        pool.clear()
    return {"history": state["history"], "reuse": state["reuse"],
            "attacker_only_complete": state["round"] + 1 == pause_after_attacker_round and state["phase"] == "generate_defender",
            "pilot_complete": state["round"] < rounds}


def run(stage, stop=None):
    plan = load_plan()
    if stage == "comparison_baseline":
        payload = baseline(plan, stop=stop)
    elif stage == "comparison_warmup":
        payload = warmup(plan, stop=stop)
    elif stage.startswith("comparison_rl_"):
        group = stage.removeprefix("comparison_rl_")
        from .readiness import require_readiness
        require_readiness(OUTPUT, GROUPS[group])
        payload = standalone(plan, group, stop=stop)
    elif stage.startswith("comparison_race_"):
        group = stage.removeprefix("comparison_race_")
        from .readiness import require_readiness
        require_readiness(OUTPUT, GROUPS[group])
        payload = race(plan, group, stop=stop)
    else:
        raise ValueError(stage)
    write_json(OUTPUT / f"{stage}_measurements.json", payload)
    return payload
