"""Fixed-subset measurements with isolated sampling state."""

from pathlib import Path
import time

import numpy as np
import torch

from ..checkpointing import rng_state, restore_rng
from .protocol import evaluate_pool, write_json


def log_training_diagnostic(attacker, step, *, subset_size=None, batch_size=8):
    from .workflow import OUTPUT
    import json

    root = Path(attacker.config.output_dir)
    primary_size = getattr(attacker, "diagnostic_samples", 32)
    subset_size = primary_size if subset_size is None else subset_size
    if subset_size < 1 or batch_size < 1:
        raise ValueError("Diagnostic sample and batch sizes must be positive")
    directory = root / ("diagnostics" if subset_size == primary_size else f"diagnostics_{subset_size}") / f"step_{step:06d}"
    result_path = directory / "metrics.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["num_generated"] != subset_size:
            raise ValueError("Cached diagnostics use a different sample count")
        return result
    base = json.loads((OUTPUT / "base_pairs.json").read_text())["evaluation"][:subset_size]
    if len(base) != subset_size:
        raise ValueError("Diagnostic sample count exceeds available evaluation examples")
    fitting = set(attacker.prompts)
    if not base or any(row["prompt"] in fitting for row in base):
        raise ValueError("Training diagnostic prompts must be excluded from attacker fitting")
    stop = getattr(attacker, "_stop_request", None)
    if stop is not None and stop.requested:
        return None
    started = time.perf_counter()
    state = rng_state()
    training = attacker.model.training
    padding_side = attacker.tokenizer.padding_side
    device = next(attacker.model.parameters()).device
    signature = {"step": step, "sample_ids": [row["sample_id"] for row in base], "seed": 42,
                 "max_new_tokens": attacker.config.max_new_tokens,
                 "max_seq_length": attacker.config.max_seq_length,
                 "temperature": attacker.config.temperature, "top_p": attacker.config.top_p,
                 "defenders": attacker.pool.checkpoints, "batch_size": batch_size,
                 "logging_only": True}
    generation_path = directory / "generations.json"
    generated = json.loads(generation_path.read_text()) if generation_path.exists() else {"signature": signature, "texts": []}
    if generated["signature"] != signature:
        raise ValueError("Cached training diagnostics refer to different inputs")
    prefix_path = root / "diagnostics" / f"step_{step:06d}/generations.json"
    if subset_size > primary_size and not generated["texts"] and prefix_path.exists():
        prefix = json.loads(prefix_path.read_text())
        prefix_signature = {**signature, "sample_ids": signature["sample_ids"][:primary_size]}
        if prefix["signature"] != prefix_signature:
            raise ValueError("Diagnostic prefix refers to different inputs")
        complete_batches = min(len(prefix["texts"]), primary_size) // batch_size * batch_size
        generated["texts"] = prefix["texts"][:complete_batches]
    try:
        attacker.model.eval()
        attacker.tokenizer.padding_side = "left"
        for start in range(len(generated["texts"]), len(base), batch_size):
            if stop is not None and stop.requested:
                return None
            torch.manual_seed(42 + start)
            tokens = attacker.tokenizer([row["prompt"] for row in base[start:start + batch_size]],
                                        return_tensors="pt", padding=True, truncation=True,
                                        max_length=attacker.config.max_seq_length).to(device)
            with torch.no_grad():
                output = attacker.model.generate(**tokens, max_new_tokens=attacker.config.max_new_tokens,
                                                  do_sample=True, temperature=attacker.config.temperature,
                                                  top_p=attacker.config.top_p,
                                                  pad_token_id=attacker.tokenizer.pad_token_id)
            generated["texts"].extend(attacker.tokenizer.batch_decode(
                output[:, tokens["input_ids"].shape[1]:], skip_special_tokens=True))
            write_json(generation_path, generated)
        pairs = [{**row, "ai": text, "eligible": row["eligible"] and len(text.split()) >= 20}
                 for row, text in zip(base, generated["texts"])]
        records = evaluate_pool(attacker.pool, pairs, directory / "scores")
        encoder = attacker.reward_fn.semantic
        if hasattr(encoder, "batch"):
            semantic = encoder.batch([row["reference"] for row in pairs], [row["ai"] for row in pairs])
        else:
            semantic = [encoder(row["reference"], row["ai"]) for row in pairs]
        value = {**signature, "records": records, "num_generated": len(pairs),
                 "mean_semantic_similarity": float(np.mean(semantic)),
                 "elapsed_seconds": time.perf_counter() - started}
        write_json(result_path, value)
        print(f"Training diagnostic step {step}: {len(pairs)} fixed prompts; "
              f"{[(row['detector'], row['attack_success_rate']) for row in records]}", flush=True)
        return value
    finally:
        attacker.tokenizer.padding_side = padding_side
        attacker.model.train(training)
        restore_rng(state)
