"""Immutable inputs and metrics for the watermarking experiments."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from ..rl_evasion.image_evasion.research import digest, file_hash, write_json


STAGES = {
    "preflight": [], "textseal": ["preflight"], "distill": ["preflight"], "distilled": ["distill"],
    "text_base": ["distilled"], "text_clean": ["distilled"],
    "rar": ["preflight"], "rar_base": ["rar"], "rar_clean": ["rar"],
    "dcae": ["preflight"], "dcae_base": ["dcae"], "dcae_clean": ["dcae"],
}
DEFAULT_STAGES = tuple(STAGES)
LATENT_STAGES = {
    "latent_preflight": [], "latent_embedder": ["latent_preflight"],
    "latent_teacher": ["latent_embedder"], "latent_distill": ["latent_teacher"],
    "latent_evaluate": ["latent_distill"], "latent_base": ["latent_evaluate"],
    "latent_clean": ["latent_evaluate"],
}
STAGES.update(LATENT_STAGES)


def use_sources(manifest):
    for name in ("distseal", "textseal"):
        path = manifest["resources"]["repositories"][name]["path"]
        sys.path.insert(0, path)
        if name == "distseal":
            sys.path.insert(0, str(Path(path) / "deps"))


def prepare_data(repository, destination, model, seed):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    groups = {"train": [], "validation": [], "test": [], "removal": []}
    seen = set()
    with (repository / "data/hc3-all.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            group = int(digest(row["question"].strip().lower())[:12], 16) % 100
            split = "train" if group < 65 else "removal" if group < 85 else "validation" if group < 90 else "test"
            for answer in row["human_answers"]:
                tokens = tokenizer.encode(answer, add_special_tokens=False)
                if len(tokens) < 320 or digest(tokens[:320]) in seen:
                    continue
                seen.add(digest(tokens[:320]))
                groups[split].append({"id": digest(row["question"]), "tokens": tokens[:512]})
                break
    for split, count in {"train": 1000, "removal": 200, "validation": 128, "test": 256}.items():
        if len(groups[split]) < count:
            raise ValueError(f"Insufficient independent HC3 {split} documents: {len(groups[split])}")
    rng = np.random.default_rng(seed)
    for split in groups:
        rng.shuffle(groups[split])
    groups["validation"] = groups["validation"][:256]
    groups["test"] = groups["test"][:256]
    imagenet = Path("/lustre/fsmisc/dataset/imagenet")
    classes = sorted(p.name for p in (imagenet / "train").iterdir() if p.is_dir())
    if len(classes) != 1000:
        raise ValueError("ImageNet class ordering must contain exactly 1000 synsets")
    # Match DistSeal's single-class forgetting setting, using training images only.
    train = sorted((imagenet / "train" / classes[0]).glob("*.JPEG"))
    validation = sorted((imagenet / "val" / classes[0]).glob("*.JPEG"))
    if len(train) < 50 or len(validation) < 40:
        raise ValueError("ImageNet class zero inputs are incomplete")
    images = {"train": [str(p) for p in rng.choice(train, 50, replace=False)],
              "validation": [str(p) for p in validation[:20]],
              "test": [str(p) for p in validation[20:40]], "synsets": classes,
              "class_zero": classes[0], "source": str(imagenet)}
    data = {"text": groups, "images": images, "seed": seed}
    write_json(destination, data)
    return {split: len(rows) for split, rows in groups.items()}


def validate_partitions(data):
    sets = [set(row["id"] for row in rows) for rows in data["text"].values()]
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            if a & b:
                raise ValueError("Text document leakage between partitions")
    sets = [set(data["images"][name]) for name in ("train", "validation", "test")]
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            if a & b:
                raise ValueError("Image leakage between partitions")


def operating_point(null_scores, fpr=0.01):
    scores = np.asarray(null_scores, dtype=float)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration scores")
    return float(np.quantile(scores, 1 - fpr, method="higher"))


def detection_metrics(scores, null_scores, threshold):
    from sklearn.metrics import roc_auc_score
    scores, null_scores = np.asarray(scores), np.asarray(null_scores)
    from scipy.stats import beta
    def rate(values):
        n, k = len(values), int((values > threshold).sum())
        return {"rate": k / n, "count": k, "n": n,
                "ci95": [float(beta.ppf(.025, k, n-k+1)) if k else 0.,
                         float(beta.ppf(.975, k+1, n-k)) if k < n else 1.]}
    return {"detection": rate(scores), "false_positive": rate(null_scores),
            "threshold": threshold,
            "auroc": float(roc_auc_score([1]*len(scores)+[0]*len(null_scores),
                                        np.r_[scores, null_scores]))}
