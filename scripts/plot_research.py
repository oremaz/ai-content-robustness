"""Render recorded research measurements as portable figures and source CSVs."""

import csv
import json
import math
from pathlib import Path
import textwrap


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _save(fig, directory, name, rows):
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / name
    for extension in ("png", "svg"):
        target = stem.with_suffix(f".{extension}")
        temporary = target.with_suffix(f".{extension}.tmp")
        fig.savefig(temporary, format=extension, dpi=170, bbox_inches="tight")
        temporary.replace(target)
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    from matplotlib import pyplot as plt
    plt.close(fig)
    return stem.with_suffix(".png")


def lines(directory, name, title, xlabel, ylabel, series, bounded=False, threshold=None):
    from matplotlib import pyplot as plt
    fig, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    rows = []
    styles = [("o", "-"), ("s", "--"), ("^", "-.")]
    for index, (label, xs, ys) in enumerate(series):
        if not any(_finite(y) for y in ys):
            continue
        marker, style = styles[index % len(styles)]
        values = [float(y) if _finite(y) else float("nan") for y in ys]
        axis.plot(xs, values, label=label, marker=marker if len(xs) <= 30 else None,
                  linestyle=style, linewidth=1.8, markersize=4)
        rows.extend({"series": label, "x": x, "y": y if _finite(y) else ""}
                    for x, y in zip(xs, ys))
    if not rows:
        plt.close(fig)
        return None
    if threshold is not None:
        axis.axhline(threshold, color="black", linestyle=":", label=f"Target {threshold:g}")
    if bounded:
        axis.set_ylim(0, 1)
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    return _save(fig, directory, name, rows)


def bars(directory, name, title, ylabel, groups):
    from matplotlib import pyplot as plt
    labels = list(dict.fromkeys(label for _, values in groups for label in values))
    finite = [float(value) for _, values in groups for value in values.values() if _finite(value)]
    if not labels or not finite:
        return None
    fig, axis = plt.subplots(figsize=(11, max(4, len(labels) * 0.65)), layout="constrained")
    height = 0.8 / len(groups)
    rows = []
    for index, (group, values) in enumerate(groups):
        ys = [position + (index - (len(groups) - 1) / 2) * height for position in range(len(labels))]
        scores = [float(values[label]) if _finite(values.get(label)) else float("nan") for label in labels]
        axis.barh(ys, scores, height=height, label=group)
        rows.extend({"series": group, "label": label, "value": score if _finite(score) else ""}
                    for label, score in zip(labels, scores))
    axis.set_yticks(range(len(labels)), [textwrap.fill(label, 34) for label in labels])
    axis.invert_yaxis()
    axis.set(title=title, xlabel=ylabel, xlim=(min(0, min(finite)) * 1.05, max(1, max(finite)) * 1.05))
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", fontsize=9)
    return _save(fig, directory, name, rows)


def stage_figures(root, stage, payload=None):
    import matplotlib
    matplotlib.use("Agg")
    root = Path(root)
    directory = root / "docs/figures/research"
    figures = []

    def add(path, caption):
        if path is not None:
            figures.append((path, caption))

    if stage in {"text_notebook", "image_notebook"}:
        add(bars(directory, stage + "-transfer", "Detector transfer across evaluation datasets", "Macro F1 (higher is better)",
                 [(name, {row["model"]: row.get("f1_macro") for row in rows})
                  for name, rows in payload["tables"].items()]),
            "Each bar is one recorded evaluation. Dataset comparisons include changes in domain and generators.")
        benchmark = root / "results" / ("bench_aitextdetect" if stage == "text_notebook" else "bench_imai_artifact")
        index_path = benchmark / "index.jsonl"
        expected = {row["model"] for rows in payload["tables"].values() for row in rows}
        if index_path.exists():
            for entry in map(json.loads, index_path.read_text().splitlines()):
                if entry["model_name"] not in expected:
                    continue
                checkpoint_id = entry["checkpoint_id"]
                native = list((benchmark / ".restart" / checkpoint_id).glob("checkpoint-*/trainer_state.json"))
                if native:
                    latest = max(native, key=lambda path: int(path.parent.name.split("-")[-1]))
                    history = json.loads(latest.read_text())["log_history"]
                    series = [("Logged training loss", [row["step"] for row in history if "loss" in row],
                               [row["loss"] for row in history if "loss" in row])]
                    xlabel = "Optimizer update"
                else:
                    metrics = benchmark / f"{checkpoint_id}_metrics.csv"
                    history = list(csv.DictReader(metrics.open())) if metrics.exists() else []
                    series = [(key, list(range(1, len(history) + 1)), [row.get(key) for row in history])
                              for key in ("loss", "val_loss")]
                    xlabel = "Epoch"
                add(lines(directory, f"{stage}-{checkpoint_id}-loss", entry["model_name"], xlabel, "Loss", series),
                    "Recorded training loss; validation loss appears only when available. Different objectives have different loss scales.")
    elif stage == "benchmark":
        for track in ("hc3", "cnn_dailymail"):
            value = json.loads((root / "results/research_workflow/benchmark" / track / "benchmark_results.json").read_text())
            rows = {name: metrics for name, metrics in value["results"].items() if metrics.get("status") == "completed"}
            add(bars(directory, f"benchmark-{track}", f"Text evasion: {track}", "Score (higher is better)",
                     [(key.replace("_", " "), {name: row.get(key) for name, row in rows.items()})
                      for key in ("mean_evasion", "mean_semantic_similarity", "attack_success_rate")]),
                "Only completed methods are plotted; the accompanying table retains failures and sample counts. Evasion is not calibrated authorship probability.")
    elif stage == "arms_race":
        history = json.loads((root / "results/arms_race/text/arms_race_history.json").read_text())
        add(lines(directory, "arms-race-paired", "Attacker and defender: paired held-out measurements", "Round", "Attacker success rate (higher favors attacker)",
                  [(label, [row["round"] for row in history], [row[key]["attack_success_rate"] for row in history])
                   for label, key in [("Before attacker update", "before_attacker_update"),
                                      ("After attacker update", "after_attacker_update"),
                                      ("After defender update", "after_defender_update")]], bounded=True),
            "Within each round the measurements share evaluation prompts and sampling seeds. Lower success after defense means recovery; ten rounds do not establish equilibrium.")
    elif stage == "meta_adapt":
        value = json.loads((root / "results/meta_adapt/adaptation_comparison.json").read_text())
        add(lines(directory, "maml-adaptation", "Adaptation to the held-out detector", "Inner adaptation update", "Mean evasion score (higher is better)",
                  [(label, value[key]["steps"], value[key]["evasion"])
                   for label, key in [("Meta-trained initialization", "meta"), ("Pre-meta initialization", "pre_meta")]],
                  bounded=True, threshold=0.7),
            "Both initializations use the same adaptation and reserved evaluation protocol. Dots are measured updates; lines are visual guides. No uncertainty interval was estimated.")
    elif stage in {"grpo_text", "ddpo_image"}:
        folder = "text_evasion/grpo" if stage == "grpo_text" else "image_evasion/ddpo"
        value = json.loads((root / "results" / folder / "evaluation.json").read_text())
        add(bars(directory, stage + "-evaluation", "Held-out evaluation: " + stage, "Recorded mean score",
                 [("Evaluation", {key: val for key, val in value.items()
                                  if key.startswith(("mean_", "evasion_")) or key == "attack_success_rate"})]),
            "These are aggregate evaluation scores, not a time curve or a comparison with an untrained generator.")

    folder = {"grpo_text": "text_evasion/grpo", "multispin": "text_evasion/multispin",
              "ddpo_image": "image_evasion/ddpo", "meta_adapt": "meta_adapt"}.get(stage)
    history_path = root / "results" / folder / "training_history.json" if folder else None
    if history_path is not None and history_path.exists():
        history = json.loads(history_path.read_text())
        specifications = {
            "grpo_text": [("step", "loss", "GRPO loss"), ("step", "reward", "Training reward")],
            "multispin": [("global_step", "spin_loss", "SPIN minibatch loss")],
            "ddpo_image": [("epoch", "mean_reward", "Sampled training reward"),
                           ("epoch", "ppo_loss", "DDPO policy loss"), ("epoch", "approx_kl", "Approximate KL penalty")],
            "meta_adapt": [("outer_step", "meta_loss", "MAML outer objective")],
        }[stage]
        for xkey, key, label in specifications:
            rows = [row for row in history if xkey in row and key in row]
            add(lines(directory, f"{stage}-{key}", label, xkey.replace("_", " ").capitalize(), label,
                      [(label, [row[xkey] for row in rows], [row[key] for row in rows])]),
                "Recorded training measurements without smoothing. Training reward or loss does not replace held-out evaluation.")
    return figures
