"""Report warm-up quality or enforce a user-defined validation target."""

import json
from pathlib import Path

import numpy as np

from .protocol import validation_threshold, write_json


def require_readiness(root, names):
    root = Path(root)
    target_path = root / "readiness_target.json"
    if not target_path.exists():
        raise RuntimeError("Warm-up readiness needs the user's minimum detection rate, maximum false-positive "
                           "rate and ensemble policy before RL can start")
    target = json.loads(target_path.read_text())
    advisory = target.get("mode") == "report_only"
    if not advisory:
        minimum, maximum = target["minimum_tpr"], target["maximum_fpr"]
        if not 0 <= minimum <= 1 or not 0 <= maximum < 1:
            raise ValueError("Readiness targets must be fractions in [0, 1]")
    scores = json.loads((root / "validation_scores.json").read_text())
    n = len(next(iter(scores.values()))) // 2
    if not n or any(len(values) != 2 * n for values in scores.values()):
        raise ValueError("Balanced validation scores are required for readiness")
    series = {name: np.asarray(values) for name, values in scores.items()}
    if any(not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
           for values in series.values()):
        raise ValueError("Readiness scores must be finite probabilities")
    series["ensemble"] = np.mean(list(series.values()), axis=0)
    report = {}
    for name, values in series.items():
        threshold = 0.5 if advisory else validation_threshold(values[n:], maximum)
        tpr = float(np.mean(values[:n] >= threshold))
        fpr = float(np.mean(values[n:] >= threshold))
        report[name] = {"tpr": tpr, "fpr": fpr, "threshold": threshold,
                        "ready": None if advisory else tpr >= minimum and fpr <= maximum}
    write_json(root / "readiness_report.json", {"target": target, "validation": report,
                                               "quality_gate_enforced": not advisory})
    if advisory:
        return report
    required = list(scores) + ["ensemble"] if names is None and target["block_ensemble_on_weak_member"] else (
        ["ensemble"] if names is None else list(names))
    failed = [name for name in required if not report[name]["ready"]]
    if failed:
        raise RuntimeError(f"Warm-up validation target not met: {failed}; see readiness_report.json")
    return report
