"""Select notebook defender families from completed MAGE measurements."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def select_notebook_defenders(payload, configurations, metric="f1_macro"):
    rows = payload["tables"]["MAGE test"]
    by_name = {row["model"]: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("Duplicate MAGE model measurements")
    expected = {"tfidf_logreg", "tfidf_random_forest", "tfidf_xgboost", "tfidf_lightgbm", *configurations}
    if set(by_name) != expected:
        raise ValueError("Complete MAGE evaluation is required before selecting defenders")
    if any(not isinstance(row.get(metric), (int, float)) or not math.isfinite(row[metric])
           for row in rows):
        raise ValueError("Selection needs finite MAGE scores for every model")
    encoder = "qlora_modernbert_large"
    if encoder not in configurations:
        raise ValueError("ModernBERT-large is missing")
    decoder_candidates = [name for name, configuration in configurations.items()
                          if "modernbert" not in configuration["hf_model"].lower()
                          and "qwen" not in configuration["hf_model"].lower()]
    if not decoder_candidates:
        raise ValueError("No non-Qwen decoder classifier was evaluated")

    def best(names):
        return sorted(names, key=lambda name: (-by_name[name][metric], name))[0]

    selected = {"tfidf": best(name for name in expected if name.startswith("tfidf_")),
                "encoder": encoder, "decoder": best(decoder_candidates)}
    return {"selection_dataset": "MAGE test", "metric": metric,
            "tie_break": "ascending model name", "selected": selected,
            "selected_scores": {role: by_name[name][metric] for role, name in selected.items()},
            "decoder_candidates": sorted(decoder_candidates), "excluded_family": "Qwen",
            "selection_is_not_independent_evaluation": True}


def save_notebook_selection(measurements_path, notebook_path, destination):
    from scripts.prepare_research import text_configs
    from ml_pipeline.utils.utils import _load_index

    measurements_path, notebook_path, destination = map(Path, (measurements_path, notebook_path, destination))
    payload = json.loads(measurements_path.read_text())
    selection = select_notebook_defenders(payload, text_configs(json.loads(notebook_path.read_text())))
    entries = list(_load_index("bench_aitextdetect").values())
    selection["checkpoints"] = {}
    for role, name in selection["selected"].items():
        matches = [entry for entry in entries if entry["model_name"] == name
                   and entry.get("augmentation_name") == "none"]
        if len(matches) != 1:
            raise ValueError(f"Expected one exact checkpoint for {name}, found {len(matches)}")
        selection["checkpoints"][role] = matches[0]
    selection["measurements_sha256"] = hashlib.sha256(measurements_path.read_bytes()).hexdigest()
    selection["notebook_job_id"] = payload["job_id"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    return selection
