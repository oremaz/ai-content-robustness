"""Readable tables and curves for the matched seven-defender experiment."""

import json

from scripts.plot_research import bars, lines
from scripts.report_research import table


def render_stage(root, stage, payload=None):
    directory = root / "results/research_workflow/defender_comparison"
    if payload is None:
        payload = json.loads((directory / f"{stage}_measurements.json").read_text())
    figures = root / "docs/figures/research"
    parts = []

    def include(path, caption):
        if path is not None:
            parts.extend([f"![{caption}]({path.relative_to(root / 'docs')})", "", caption,
                          f"[Source CSV]({path.with_suffix('.csv').relative_to(root / 'docs')}) · "
                          f"[Vector SVG]({path.with_suffix('.svg').relative_to(root / 'docs')})", ""])

    if "records" in payload:
        columns = ["detector", "accuracy", "f1_macro", "roc_auc", "ai_detection_rate",
                   "human_false_positive_rate", "attack_success_rate", "num_ai", "num_ineligible_pairs"]
        if stage == "comparison_warmup" or stage.startswith("comparison_rl_"):
            columns += ["tpr_at_validation_1pct_fpr", "test_fpr_at_validation_threshold"]
        parts.extend([table(payload["records"], columns), ""])
        include(bars(figures, stage + "-detection", stage.replace("_", " "), "Measured fraction",
                     [(metric.replace("_", " "), {row["detector"]: row[metric] for row in payload["records"]})
                      for metric in ("ai_detection_rate", "human_false_positive_rate")]),
                "Detection metrics use eligible pairs. Attack success uses all generated outputs; rejected short outputs never count as successful evasion.")
        if stage == "comparison_warmup":
            previous = json.loads((directory / "comparison_baseline_measurements.json").read_text())
            include(bars(figures, "comparison-before-after-warmup", "Base-Qwen detection before and after warm-up", "AUROC",
                         [(label, {row["detector"]: row["roc_auc"] for row in value["records"]})
                          for label, value in [("Original detector", previous), ("Warmed defender", payload)]]),
                    "Both evaluations use the same eligible base-Qwen/human pairs; AUROC measures ranking.")
            for detector in (directory / "adapted_defenders").iterdir():
                checkpoints = list((detector / "warmup").glob("checkpoint-*/trainer_state.json"))
                if checkpoints:
                    checkpoint = max(checkpoints, key=lambda path: int(path.parent.name.split("-")[-1]))
                    history = json.loads(checkpoint.read_text())["log_history"]
                    series = [(label, [row["step"] for row in history if key in row],
                               [row[key] for row in history if key in row])
                              for key, label in [("loss", "Training loss"), ("eval_loss", "Validation loss")]]
                    include(lines(figures, "warmup-" + detector.name + "-loss", detector.name + " warm-up",
                                  "Optimizer update", "Cross-entropy loss", series),
                            "Native training logs; validation uses held-out warm-up prompt pairs.")
        for metric in ("mean_semantic_similarity", "mean_ai_words", "mean_human_words"):
            if metric in payload and payload[metric] is not None:
                parts.extend([f"{metric}: {payload[metric]:.4f}.", ""])
        if stage.startswith("comparison_rl_"):
            group = stage.removeprefix("comparison_rl_")
            history = json.loads((directory / "rl" / group / "training_history.json").read_text())
            for metric in ("loss", "reward"):
                rows = [row for row in history if "step" in row and metric in row]
                include(lines(figures, stage + "-" + metric, "Attacker training " + metric,
                              "Optimizer update", metric,
                              [(metric, [row["step"] for row in rows], [row[metric] for row in rows])]),
                        "Recorded training diagnostics; held-out detection remains the performance measurement.")
        if stage in {"comparison_baseline", "comparison_warmup"}:
            from sklearn.metrics import roc_curve
            artifact = json.loads((directory / ("baseline" if stage.endswith("baseline") else "warmup") /
                                   "measurements.json").read_text())
            series = []
            for name, scores in artifact["scores"].items():
                n = len(scores) // 2
                if not n:
                    continue
                fpr, tpr, _ = roc_curve([1] * n + [0] * n, scores)
                series.append((name, fpr.tolist(), tpr.tolist()))
            include(lines(figures, stage + "-roc", "Base-Qwen detector ROC curves", "Human false-positive rate",
                          "AI detection rate", series, bounded=True),
                    "Descriptive held-out ROC curves. The reported fixed operating thresholds come from validation only.")
    else:
        names = [row["detector"] for row in payload["history"][0]["before"]["records"]]
        rows = []
        for round_result in payload["history"]:
            for name in names:
                row = {"round": round_result["round"], "detector": name}
                for phase in ("before", "after_attack", "after_defense"):
                    record = next(item for item in round_result[phase]["records"] if item["detector"] == name)
                    row[phase] = record["attack_success_rate"]
                rows.append(row)
        parts.extend([table(rows, ["round", "detector", "before", "after_attack", "after_defense"]), ""])
        for name in names:
            selected = [row for row in rows if row["detector"] == name]
            include(lines(figures, stage + "-" + name, name + ": paired attacker/defender updates", "Round",
                          "Attacker success rate",
                          [(phase.replace("_", " "), [row["round"] for row in selected],
                            [row[phase] for row in selected]) for phase in ("before", "after_attack", "after_defense")],
                          bounded=True), "The after-attack and after-defense measurements reuse identical generated texts.")
        parts.extend([f"Round-one checkpoint reuse: {payload['reuse']['reason']}.", ""])
    parts.extend(["These are single-seed results. Successful warm-up and robustness must be assessed from "
                  "the measurements; neither is guaranteed by the training procedure.", ""])
    return "\n".join(parts)
