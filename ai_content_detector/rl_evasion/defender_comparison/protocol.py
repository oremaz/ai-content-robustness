"""Shared immutable examples, detector scoring and measured operating points."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np
import torch

from ..arms_race.equilibrium import ArmsRaceExperiment
from ..config import ArmsRaceConfig, TextEvasionConfig
from ..checkpointing import CheckpointPause, StopRequest, atomic_save, load_checkpoint


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def generation(attacker, prompts, directory, context, seed=42):
    batch_size = getattr(attacker, "generation_batch_size", 1)
    if batch_size > 1:
        from .generation import generate_batched
        return generate_batched(attacker, prompts, directory, context, seed, batch_size)
    experiment = ArmsRaceExperiment(ArmsRaceConfig(output_dir=str(directory)))
    experiment.attacker = attacker
    experiment._generation_context = context
    return experiment._generate(prompts, seed)


def matched_pairs(prompts, references, generated, tokenizer):
    if not (len(prompts) == len(references) == len(generated)):
        raise ValueError("Generated and human examples must be aligned")
    records = []
    for prompt, reference, output in zip(prompts, references, generated):
        tokens = tokenizer.encode(output, add_special_tokens=False)
        human_tokens = tokenizer.encode(reference, add_special_tokens=False)[:len(tokens)]
        human = tokenizer.decode(human_tokens, skip_special_tokens=True)
        records.append({"prompt": prompt, "ai": output, "human": human,
                        "reference": reference, "sample_id": hashlib.sha256(prompt.encode()).hexdigest(),
                        "ai_tokens": len(tokens), "human_tokens": len(human_tokens),
                        "eligible": len(output.split()) >= 20 and len(human.split()) >= 20})
    return records


def prepare_pairs(root, stop=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..text_evasion.grpo_trainer import GRPOTextEvasionTrainer

    root = Path(root)
    destination = root / "base_pairs.json"
    if destination.exists():
        return json.loads(destination.read_text())
    cfg = TextEvasionConfig()
    attacker = GRPOTextEvasionTrainer(cfg)
    attacker._setup_dataset()
    if set(attacker.prompts) & set(attacker.eval_prompts):
        raise ValueError("Training and evaluation prompts overlap")
    write_json(root / "prompt_manifest.json", {
        "train": attacker.prompts, "train_references": attacker.references,
        "evaluation": attacker.eval_prompts, "evaluation_references": attacker.eval_references,
        "seed": cfg.seed, "warmup_prompt_limit": 1000, "evaluation_prompt_limit": 500,
        "source": "abisee/cnn_dailymail", "configuration": "3.0.0", "split": "train",
    })
    attacker.tokenizer = AutoTokenizer.from_pretrained(cfg.generator_model, trust_remote_code=True)
    if attacker.tokenizer.pad_token is None:
        attacker.tokenizer.pad_token = attacker.tokenizer.eos_token
    attacker.model = AutoModelForCausalLM.from_pretrained(
        cfg.generator_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0",
    ).eval().requires_grad_(False)
    if stop is not None and stop.requested:
        raise CheckpointPause("Baseline model setup interrupted before generation")
    payload = {"generator": cfg.generator_model, "adaptation": "none", "seed": cfg.seed}
    for name, prompts, references in (
        ("warmup", attacker.prompts[:1000], attacker.references[:1000]),
        ("evaluation", attacker.eval_prompts[:500], attacker.eval_references[:500]),
    ):
        outputs = generation(attacker, prompts, root / "base_generation", f"base_qwen/{name}", seed=cfg.seed)
        payload[name] = matched_pairs(prompts, references, outputs, attacker.tokenizer)
    write_json(destination, payload)
    del attacker
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def score_checkpointed(pool, texts, directory):
    directory = Path(directory)
    identities = {}
    for name, checkpoint in pool.checkpoints.items():
        directory_path = Path(checkpoint)
        identities[name] = {}
        for filename in ("completed.pt", "classifier.pkl"):
            artifact = directory_path / filename
            if artifact.exists():
                with artifact.open("rb") as stream:
                    identities[name][filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    signature = {"texts": texts, "specifications": pool.specifications,
                 "checkpoints": pool.checkpoints, "identities": identities,
                 "implementation": hashlib.sha256(Path(__file__).with_name("defenders.py").read_bytes()).hexdigest()}
    path = directory / (digest(signature) + ".pt")
    state = load_checkpoint(path, signature) or {"config": signature, "scores": {}}
    with StopRequest() as stop:
        for name in pool.specifications:
            values = state["scores"].setdefault(name, [])
            try:
                for offset in range(len(values), len(texts), 8):
                    if stop.requested:
                        raise CheckpointPause("Detector evaluation scores retained")
                    scores = pool.get(name).predict_batch(texts[offset:offset + 8])
                    scores = np.asarray(scores, dtype=float)
                    if (not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
                            or len(scores) != min(8, len(texts) - offset)):
                        raise ValueError(f"Invalid predictions from {name}")
                    values.extend(scores.tolist())
                    atomic_save(state, path)
                    if stop.requested:
                        raise CheckpointPause("Detector evaluation scores saved")
            finally:
                pool.release(name)
    return state["scores"]


def classification_metrics(ai_scores, human_scores, threshold=0.5):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    ai, human = np.asarray(ai_scores), np.asarray(human_scores)
    if not len(ai) or not len(human):
        return {name: None for name in ("accuracy", "f1_macro", "roc_auc", "ai_detection_rate",
                                        "human_false_positive_rate", "attack_success_rate")} | {
            "threshold": float(threshold), "num_ai": len(ai), "num_human": len(human)}
    scores = np.concatenate([ai, human])
    labels = np.concatenate([np.ones(len(ai)), np.zeros(len(human))])
    return {"accuracy": float(accuracy_score(labels, scores >= threshold)),
            "f1_macro": float(f1_score(labels, scores >= threshold, average="macro")),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "ai_detection_rate": float(np.mean(ai >= threshold)),
            "human_false_positive_rate": float(np.mean(human >= threshold)),
            "attack_success_rate": float(np.mean(ai < threshold)),
            "threshold": float(threshold), "num_ai": len(ai), "num_human": len(human)}


def validation_threshold(human_scores, target_fpr=0.01):
    values = np.sort(np.asarray(human_scores, dtype=np.float64))[::-1]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite human validation scores are required")
    allowed = min(len(values) - 1, int(np.floor(target_fpr * len(values))))
    return float(np.nextafter(values[allowed], np.inf))


def evaluate_pool(pool, pairs, directory, thresholds=None):
    eligible = [row for row in pairs if row["eligible"]]
    texts = [row["ai"] for row in eligible] + [row["human"] for row in eligible]
    scores = score_checkpointed(pool, texts, directory)
    n = len(eligible)
    records = []
    for name, values in scores.items():
        record = {"detector": name, **classification_metrics(values[:n], values[n:]),
                  "num_generated": len(pairs), "num_ineligible_pairs": len(pairs) - n}
        record["conditional_attack_success_rate"] = record["attack_success_rate"]
        record["attack_success_rate"] = sum(value < 0.5 for value in values[:n]) / len(pairs)
        if thresholds and name in thresholds:
            operating = classification_metrics(values[:n], values[n:], thresholds[name])
            record.update({"tpr_at_validation_1pct_fpr": operating["ai_detection_rate"],
                           "test_fpr_at_validation_threshold": operating["human_false_positive_rate"],
                           "validation_threshold": thresholds[name]})
        records.append(record)
    write_json(Path(directory) / "measurements.json", {"records": records, "scores": scores,
                                                      "eligible_ids": [row["sample_id"] for row in eligible]})
    return records
