"""Publish only the retained attacker evidence and current defender updates."""

import json
import os
from pathlib import Path

from scripts.plot_research import bars, lines
from scripts.report_research import architecture_races, table


def update_retained_report(root, output, selection):
    report = root / "docs/research_results.md"
    figures = root / "docs/figures/research"
    parts = ["# Retained attacker results and defender updates", "",
             "The generator is `Qwen/Qwen3.5-9B-Base`. Three retained long-run "
             "attackers were trained against Gemma, ModernBERT-large and the seven-detector ensemble. "
             "Their step-500 checkpoints and final saved checkpoints are retained. A separate "
             "RoBERTa step-500 checkpoint is retained to document its fluctuating trajectory.", "",
             "## Retained detection results", "",
             "Evasion means the fraction of generated outputs with AI score below 0.5. "
             "The ensemble uses the arithmetic mean of its seven AI scores. "
             "The full evaluations use 500 generated texts and 500 human passages. "
             "RoBERTa's update-500 result uses a 100-prompt diagnostic subset, as labelled.", ""]
    completed = []
    for manifest_path in selection["race_manifests"]:
        manifest = json.loads(Path(manifest_path).read_text())
        for name, job in manifest["jobs"].items():
            directory = Path(job["directory"])
            receipt = directory / "receipt.json"
            if not receipt.exists() or json.loads(receipt.read_text())["status"] != "completed":
                continue
            before = json.loads((directory / "round_1/after_attack/evaluation.json").read_text())
            after = json.loads((directory / "round_1/after_defense/evaluation.json").read_text())
            initial = next(row for row in before["records"] if row["detector"] == "ensemble_mean")
            final = next(row for row in after["records"] if row["detector"] == "ensemble_mean")
            accounting = manifest.get("completion_accounting", {}).get(name, {})
            completed.append({"defender": name, "evasion_before": initial["attack_success_rate"],
                              "evasion_after": final["attack_success_rate"],
                              "AI_detection_after": final["ai_detection_rate"],
                              "human_FPR_after": final["human_false_positive_rate"],
                              "allocation_time": accounting.get("elapsed", "not recorded")})
    if completed:
        parts[4:4] = ["## Completed defender updates", "",
                      table(completed, ["defender", "evasion_before", "evasion_after", "AI_detection_after",
                                        "human_FPR_after", "allocation_time"]), "",
                      "Each comparison scores the same 500 saved attack texts and 500 human passages "
                      "before and after one defender update, at threshold 0.5. The attackers remain "
                      "unchanged. Detection largely recovers against these fixed attackers. Resistance "
                      "to another round of adaptive RL has not been tested. The per-member ensemble "
                      "tables below retain differences hidden by the aggregate score.", ""]
    rows = []
    for item in selection["retained"]:
        if item.get("evaluation"):
            value = json.loads(Path(item["evaluation"]).read_text())
            record = next(row for row in value["records"] if row["detector"] == item["detector"])
            rows.append({"target": item["label"], "updates": item["final_updates"],
                         "generated_texts": record["num_generated"],
                         "evasion": record["attack_success_rate"], "human_FPR": record["human_false_positive_rate"],
                         "semantic_similarity": value["mean_semantic_similarity"]})
        else:
            value = json.loads((Path(item["diagnostics"]) / "step_000500/metrics.json").read_text())
            record = next(row for row in value["records"] if row["detector"] == item["detector"])
            rows.append({"target": item["label"] + " (diagnostic)", "updates": 500,
                         "generated_texts": record["num_generated"],
                         "evasion": record["attack_success_rate"], "human_FPR": record["human_false_positive_rate"],
                         "semantic_similarity": value["mean_semantic_similarity"]})
    parts += [table(rows, ["target", "updates", "generated_texts", "evasion", "human_FPR", "semantic_similarity"]), "",
              "All three long runs used eight total completions per update and no TRL KL penalty. "
              "Their original learning rate started at `5e-5` and decayed over an 11,988-update horizon. "
              "Their step-500 adapters remain available, but no separate full detection evaluation "
              "was recorded exactly at 500. The ensemble's full result was measured at 502, not 500.", "",
              "Semantic similarity is E5 cosine similarity to the human reference. It does not "
              "establish factual accuracy or absence of policy drift. The ensemble text inspection "
              "found repeated attribution and some changed facts despite readable continuations.", ""]

    def include(path):
        if path is not None:
            relative = path.relative_to(report.parent)
            parts.extend([f"![{path.stem}]({relative})", "",
                          f"[Source CSV]({relative.with_suffix('.csv')}) · "
                          f"[Vector SVG]({relative.with_suffix('.svg')}).", ""])

    for item in selection["retained"]:
        parts += [f"## {item['label']}: retained checkpoints", "",
                  f"Step 500: `{item['step_500']}`.", ""]
        if item.get("final_checkpoint"):
            parts += [f"Final saved attacker: `{item['final_checkpoint']}` "
                      f"at {item['final_updates']} updates.", ""]
        path = Path(item.get("final_checkpoint", item["step_500"])) / "trainer_state.json"
        history = json.loads(path.read_text())["log_history"]
        for metric in ("reward", "loss", "learning_rate"):
            selected = [row for row in history if metric in row and "step" in row]
            include(lines(figures, f"retained-{item['name']}-{metric}", f"{item['label']}: recorded {metric}",
                          "Attacker optimizer update", metric,
                          [(item["label"], [row["step"] for row in selected], [row[metric] for row in selected])]))
        if item.get("diagnostics"):
            points = [json.loads(path.read_text()) for path in
                      sorted(Path(item["diagnostics"]).glob("step_*/metrics.json"))]
            records = [{"updates": point["step"], "generated_texts": point["num_generated"],
                        "evasion": next(row["attack_success_rate"] for row in point["records"]
                                        if row["detector"] == item["detector"])} for point in points]
            parts += [table(records, ["updates", "generated_texts", "evasion"]), ""]
            include(lines(figures, f"retained-{item['name']}-diagnostic-evasion", "RoBERTa: fixed 100-prompt diagnostics",
                          "Attacker optimizer update", "Evasion fraction",
                          [("RoBERTa", [row["updates"] for row in records], [row["evasion"] for row in records])],
                          bounded=True))
            parts += ["The RoBERTa trajectory uses a separate individual-attacker protocol: "
                      "50-update warm-up to `5e-5`, then a constant rate through 600. Evasion was "
                      "98% at 500, 93% at 550 and 36% at 600 on the same 100 prompts. "
                      "The full 500-text evaluation at 600 gave 39.2% evasion. Only the step-500 "
                      "model is retained; later measurements remain as evidence of fluctuation. "
                      "A high score at one checkpoint therefore does not establish stable convergence.", ""]
    parts += ["## Interpretation limits", "",
              "Training budgets and target detectors differ across the long runs, so their final "
              "scores are not a matched-budget architecture ranking. These are single-seed results. "
              "The diagnostic evaluation set has informed protocol choices and is not an untouched "
              "final test. The retained evaluations show evasion against frozen warmed defenders; "
              "the new jobs measure whether defender updates recover detection.", "",
              "## Current experiment: one defender update per retained attacker", "",
              "Three independent jobs generate training outputs on the existing 1,000 fitting prompts, "
              "pair them with human passages, and update the warmed defender(s). Torch defenders use "
              "the existing three-epoch supervised update and paired 80/20 training/validation split. "
              "The 500 evaluation prompts are excluded from training. The post-defense measurement "
              "reuses exactly the saved attack texts from the original evaluation. "
              "No attacker optimizer runs and no second round starts in these submissions.", ""]
    manifests = [Path(path) for path in selection["race_manifests"]]
    parts += architecture_races(report, manifest_paths=manifests)
    for manifest_path in sorted(manifests):
        manifest = json.loads(manifest_path.read_text())
        base_manifest = manifest.get("base_evaluation_manifest")
        if base_manifest:
            base_path = Path(base_manifest)
            base = json.loads(base_path.read_text())
            parts += ["## Updated defenders on the base model", "",
                      "This check reuses the original 500 reserved base-Qwen outputs and 500 human "
                      "passages. It generates no new text and performs no training. The comparison "
                      "uses threshold 0.5 before and after the defender update; the earlier model "
                      "is the Qwen-warmed defender, before adaptation to the RL attacker.", ""]
            rows, pending, summaries = [], [], []
            for name, branch in base["branches"].items():
                job = base["jobs"].get(name)
                result = Path(job["directory"]) / "evaluation.json" if job else None
                if result is None or not result.exists():
                    receipt = Path(job["directory"]) / "receipt.json" if job else None
                    status = json.loads(receipt.read_text())["status"] if receipt and receipt.exists() else "submitted" if job else "prepared"
                    pending.append({"branch": name, "job": job["job_id"] if job else "not submitted", "status": status})
                    continue
                before = {row["detector"]: row for row in branch["before"]["records"]}
                records = json.loads(result.read_text())["records"]
                aggregate = next(row for row in records if row["detector"] == "ensemble_mean")
                initial = before["ensemble_mean"]
                summaries.append(f"{name}: base-Qwen detection changes from {initial['ai_detection_rate']:.1%} "
                                 f"to {aggregate['ai_detection_rate']:.1%}; human false positives change "
                                 f"from {initial['human_false_positive_rate']:.1%} "
                                 f"to {aggregate['human_false_positive_rate']:.1%}.")
                for row in records:
                    if name != "ensemble" and row["detector"] == "ensemble_mean":
                        continue
                    initial = before[row["detector"]]
                    rows.append({"branch": name, "detector": row["detector"], "AI_texts": row["num_ai"],
                                 "AI_detection_before": initial["ai_detection_rate"],
                                 "AI_detection_after": row["ai_detection_rate"],
                                 "human_FPR_before": initial["human_false_positive_rate"],
                                 "human_FPR_after": row["human_false_positive_rate"], "AUROC_after": row["roc_auc"]})
            if rows:
                parts += [" ".join(summaries), ""]
                parts += [table(rows, ["branch", "detector", "AI_texts", "AI_detection_before", "AI_detection_after",
                                       "human_FPR_before", "human_FPR_after", "AUROC_after"]), ""]
                largest_drop = max(rows, key=lambda row: row["AI_detection_before"] - row["AI_detection_after"])
                if largest_drop["AI_detection_before"] > largest_drop["AI_detection_after"]:
                    parts += [f"The largest observed base-detection decrease is {largest_drop['branch']}/"
                              f"{largest_drop['detector']}: {largest_drop['AI_detection_before']:.1%} "
                              f"to {largest_drop['AI_detection_after']:.1%}, with post-update human "
                              f"false positives of {largest_drop['human_FPR_after']:.1%} "
                              f"and AUROC {largest_drop['AUROC_after']:.4f}. Aggregate ensemble "
                              "performance can conceal such member-level losses.", ""]
                for metric, title in (("AI_detection", "AI detection"), ("human_FPR", "Human false-positive rate")):
                    include(bars(figures, f"base-retest-{manifest_path.parent.name}-{metric}",
                                 f"Updated defenders on base Qwen: {title}", "Fraction",
                                 [(label, {f"{row['branch']}: {row['detector']}": row[f"{metric}_{stage}"] for row in rows})
                                  for stage, label in (("before", "Before defender update"), ("after", "After defender update"))]))
                parts += ["This measures retention on base outputs after learning the fixed RL attack. "
                          "A drop in base detection shows that recovering detection on the RL outputs "
                          "does not automatically preserve the original operating point. The human "
                          "false-positive rate and AUROC help distinguish that tradeoff from a broad "
                          "loss of ranking quality. This does not measure robustness to a new adaptive attacker.", ""]
            if pending:
                parts += [table(pending, ["branch", "job", "status"]), ""]
            parts += [f"[Shared base-evaluation metrics]({os.path.relpath(base['metrics_file'], report.parent)}).", ""]
        if not manifest.get("warm_starts"):
            continue
        for name, job in manifest["jobs"].items():
            for detector in manifest["protocol"]["defenders"][name]:
                directory = Path(job["directory"]) / "round_1/defenders" / detector / "round_1"
                states = sorted(directory.glob("checkpoint-*/trainer_state.json"),
                                key=lambda path: int(path.parent.name.split("-")[-1]))
                if not states:
                    continue
                logs = json.loads(states[-1].read_text())["log_history"]
                series = [(label, [row["step"] for row in logs if metric in row],
                           [row[metric] for row in logs if metric in row])
                          for metric, label in (("loss", "Training"), ("eval_loss", "Validation"))]
                include(lines(figures, f"defense-{manifest_path.parent.name}-{name}-{detector}-loss",
                              f"{name}: {detector} defender update", "Defender optimizer update", "Supervised loss", series))
    report.write_text("\n".join(parts).rstrip() + "\n")
    return report
