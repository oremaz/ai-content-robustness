"""Paired semantic similarity for frozen watermark experiment outputs."""

import hashlib

import numpy as np


def text_key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def aligned_rows(original, candidate):
    ids = [row["id"] for row in original]
    by_id = {row["id"]: row for row in candidate}
    if len(set(ids)) != len(ids) or len(by_id) != len(candidate):
        raise ValueError("Duplicate prompt IDs in paired evaluation")
    if set(ids) != set(by_id):
        raise ValueError("Candidate prompt IDs differ from the original responses")
    return [by_id[key] for key in ids]


def last_token_pool(hidden, attention_mask):
    import torch
    if hidden.shape[:2] != attention_mask.shape or not attention_mask.any(-1).all():
        raise ValueError("Invalid embedding attention mask")
    indices = (torch.arange(hidden.shape[1], device=hidden.device)[None]*attention_mask).max(-1).values
    return hidden[torch.arange(len(hidden), device=hidden.device), indices]


def cosine_pairs(first, second):
    first, second = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    if first.ndim != 2 or first.shape != second.shape:
        raise ValueError("Paired embeddings must have matching matrix shapes")
    norms = np.linalg.norm(first, axis=1)*np.linalg.norm(second, axis=1)
    if not np.isfinite(first).all() or not np.isfinite(second).all() or np.any(norms <= 0):
        raise ValueError("Embedding vectors must be finite and nonzero")
    return np.clip(np.einsum("ij,ij->i", first, second)/norms, -1, 1)


def summarize(values, seed=20261104):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Similarity summary requires finite observations")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return {"n": len(values), "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values)>1 else 0.,
            "median": float(np.median(values)), "p05": float(np.quantile(values, .05)),
            "p25": float(np.quantile(values, .25)), "p75": float(np.quantile(values, .75)),
            "p95": float(np.quantile(values, .95)), "mean_ci95": np.quantile(means, [.025, .975]).tolist()}


def paired_scores(original, candidate, vectors):
    candidate = aligned_rows(original, candidate)
    first = np.stack([vectors[text_key(row["text"])] for row in original])
    second = np.stack([vectors[text_key(row["text"])] for row in candidate])
    values = cosine_pairs(first, second)
    return [{"id": a["id"], "cosine": float(value), "original_sha256": text_key(a["text"]),
             "candidate_sha256": text_key(b["text"]), "empty": not a["text"].strip() or not b["text"].strip()}
            for a, b, value in zip(original, candidate, values)]
