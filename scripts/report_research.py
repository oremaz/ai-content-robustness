"""Publish measurements from completed research stages without historical outputs."""

import json
import fcntl
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/research_workflow"
MARKER = "<!-- completed-research-measurements -->"


def table(records, columns):
    def cell(value):
        if value is None or isinstance(value, float) and not math.isfinite(value):
            return "undefined"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(cell(row.get(key)) for key in columns) + " |" for row in records),
    ])


def notebook_analysis(payload, level=2):
    parts = [f"{'#' * level} Analysis and Lessons Learned", "",
             f"Slurm job `{payload['job_id']}`. These measurements belong to this execution.", ""]
    for name, rows in payload["tables"].items():
        parts += [f"{'#' * (level + 1)} {name}", "", table(rows, ["model", "accuracy", "f1_macro", "roc_auc"]), ""]
        finite = [row for row in rows if row.get("f1_macro") is not None]
        if finite:
            best = max(finite, key=lambda row: row["f1_macro"])
            parts += [f"The highest observed macro F1 is {best['f1_macro']:.4f}, from `{best['model']}`.", ""]
    parts += ["Macro F1 weights the two classes equally. Compare each model across datasets: "
              "an external drop measures transfer on these samples, while a high random holdout "
              "score alone does not establish robustness to a new generator or acquisition pipeline. "
              "These are single-run comparisons without uncertainty intervals.", ""]
    if "intervention" in payload:
        parts += [f"{'#' * (level + 1)} ConvNeXt intervention", "", table(payload["intervention"],
                  ["dataset", "variant", "accuracy", "f1_macro", "f1_delta"]), "",
                  f"The predefined exploratory gate {'passed' if payload['passes_exploratory_gate'] else 'failed'}. "
                  "It requires positive macro-F1 changes on both external datasets and an ArtiFact "
                  "loss of at most 0.03. Because those external sets informed the intervention, "
                  "a gate pass still requires an untouched confirmation dataset.", ""]
    else:
        parts += ["The peer-review evaluation combines a new domain with newer generators; "
                  "it does not isolate a temporal effect. Detector scores are comparative signals, "
                  "not calibrated authorship probabilities.", ""]
    return "\n".join(parts)


def finalize_notebook(stage, executed_path):
    import nbformat

    payload = json.loads((OUTPUT / f"{stage}_measurements.json").read_text())
    summary = notebook_analysis(payload)
    executed = nbformat.read(executed_path, as_version=4)
    source_path = ROOT / "ai_content_detector/notebooks" / (
        "bench-aitextdetect.ipynb" if stage == "text_notebook" else "bench-imai-artifact.ipynb"
    )
    source = nbformat.read(source_path, as_version=4)
    for index, cell in enumerate(source.cells):
        if cell.cell_type == "markdown" and "Analysis and Lessons Learned" in cell.source:
            cell.source = summary
            executed.cells[index].source = summary
        elif cell.cell_type == "markdown" and "original experiment is intentionally retained" in cell.source and not cell.source.startswith("Historical motivation:"):
            cell.source = "Historical motivation: the numerical values below belong to an earlier execution. " \
                          "Current measurements and the intervention decision appear in the final analysis.\n\n" + cell.source
            executed.cells[index].source = cell.source
        elif cell.cell_type == "code":
            cell.outputs = executed.cells[index].outputs
            cell.execution_count = executed.cells[index].execution_count
    nbformat.write(source, source_path)
    nbformat.write(executed, executed_path)


def update_report():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _update_report()


def checkpoint_evaluations(report):
    from scripts.plot_research import bars, lines

    root = OUTPUT / "defender_comparison"
    parts = []
    for path in sorted((root / "checkpoint_evaluations").glob("*/*/results.json")):
        value = json.loads(path.read_text())
        metadata = value["checkpoint"]
        group, step = metadata["group"], metadata["global_step"]
        parts += [f"### Paused attacker evaluation: {group}, update {step}", "",
                  f"Saved checkpoint: `{metadata['checkpoint']}`. The original schedule retains "
                  f"{metadata['scheduled_steps']} updates. No further training was performed for this evaluation.", "",
                  table(value["records"], ["detector", "attack_success_rate", "ai_detection_rate",
                        "human_false_positive_rate", "roc_auc", "tpr_at_validation_1pct_fpr",
                        "test_fpr_at_validation_threshold", "num_generated", "num_ineligible_pairs"]), "",
                  table([value], ["mean_semantic_similarity", "mean_ai_words", "mean_human_words"]), "",
                  "Attack success uses all generated outputs as its denominator; other detection metrics "
                  "use eligible pairs. Each branch is evaluated against its own training target(s). "
                  "Different checkpoint steps do not form a matched-budget architecture comparison. "
                  "These held-out results can inform a stopping decision, but that decision makes this "
                  "set a model-selection set for subsequent work.", ""]
        figures = ROOT / "docs/figures/research"
        prefix = f"paused-{group}-{step}"
        baseline = json.loads((root / "warmup/measurements.json").read_text())["records"]
        names = {row["detector"] for row in value["records"] if row["detector"] != "ensemble_mean"}
        plots = [bars(figures, prefix + "-evasion", "Held-out evasion against warmed defenders",
                      "Attack success rate (higher favors attacker)", [
                          ("Base Qwen", {row["detector"]: row["attack_success_rate"] for row in baseline
                                         if row["detector"] in names}),
                          (f"RL update {step}", {row["detector"]: row["attack_success_rate"]
                                                for row in value["records"] if row["detector"] in names})])]
        measurements = json.loads((path.parent / "scores/measurements.json").read_text())
        from sklearn.metrics import roc_curve
        curves = []
        for name, scores in measurements["scores"].items():
            n = len(scores) // 2
            if n:
                fpr, tpr, _ = roc_curve([1] * n + [0] * n, scores)
                curves.append((name, fpr.tolist(), tpr.tolist()))
        plots.append(lines(figures, prefix + "-roc", "Paused attacker: held-out detection ROC",
                           "Human false-positive rate", "AI detection rate", curves, bounded=True))
        history = json.loads((Path(metadata["checkpoint"]) / "trainer_state.json").read_text())["log_history"]
        rewards = [row for row in history if "reward" in row and "step" in row]
        plots.append(lines(figures, prefix + "-reward", "Recorded attacker training reward",
                           "Optimizer update", "Training reward", [
                               (group, [row["step"] for row in rewards], [row["reward"] for row in rewards])]))
        for figure in plots:
            if figure is not None:
                relative = figure.relative_to(report.parent)
                parts += [f"![{figure.stem}]({relative})", "",
                          f"[Source CSV]({relative.with_suffix('.csv')}) · "
                          f"[Vector SVG]({relative.with_suffix('.svg')}).", ""]
    return parts


def architecture_races(report, manifest_paths=None):
    from scripts.report_defender_comparison import render_stage
    from scripts.plot_research import lines

    parts = []
    paths = ((OUTPUT / "defender_comparison/races_100").glob("*/manifest.json")
             if manifest_paths is None else manifest_paths)
    for path in sorted(paths):
        manifest = json.loads(path.read_text())
        protocol = manifest["protocol"]
        retained = manifest.get("warm_starts", {})
        description = manifest.get("historical_protocol_note",
                                   f"{protocol['attacker_steps_per_round']} attacker optimizer updates per round. ")
        initialization = "Each race starts from base Qwen and its selected warmed defender. "
        if retained:
            description = "One defender update per branch; zero new attacker updates. "
            initialization = "Each branch starts from its retained attacker and the original warmed defender(s). "
        if manifest.get("pause_after_round"):
            description += f"Pilot pauses after round {manifest['pause_after_round']}. "
        if manifest.get("pause_after_attacker_round"):
            description += "Attacker-only pilot: stop after the first post-attack evaluation, before defender training. "
        parts += [f"## Architecture races: {path.parent.name}", "",
                  f"{protocol['rounds']} planned rounds. " + description +
                  initialization +
                  f"Allocation: {manifest.get('time_limit', '20:00:00')}; QoS: {manifest.get('qos', 'qos_gpu_h100-t3')}. "
                  "Checkpoint pauses requeue the same job.", ""]
        training = protocol.get("execution", {}).get("attacker_training")
        if training:
            parts += [f"Attacker learning rate: `{training['learning_rate']}`; scheduler: "
                      f"`{training['lr_scheduler_type']}`; warm-up: {training['warmup_steps']} optimizer updates.", ""]
            if training.get("grpo_beta", 0):
                effective = training["per_device_train_batch_size"] * training["gradient_accumulation_steps"]
                generations = training["grpo_num_generations"]
                parts += [f"TRL KL coefficient: **{training['grpo_beta']}**, with a frozen initial reference adapter. "
                          f"Effective batch: **{effective} completions per optimizer update** = "
                          f"{effective // generations} prompt groups × {generations} completions. "
                          "The physical microbatch is profiled once on the allocated GPU and retained on restart.", ""]
            if training["lr_scheduler_type"] == "warmup_stable_decay":
                kwargs = training["lr_scheduler_kwargs"]
                start = protocol["attacker_steps_per_round"] - kwargs["num_decay_steps"]
                parts += [f"Linear warm-up starts at zero; the rate stays constant through update {start}, "
                          f"then follows a cosine decay to `{training['learning_rate'] * kwargs['min_lr_ratio']:.1e}` "
                          f"at update {protocol['attacker_steps_per_round']} in each attacker phase.", ""]
        if manifest.get("continuation"):
            continuation = manifest["continuation"]
            source = Path(continuation["source_manifest"]).relative_to(ROOT)
            label = "Dependent continuation from the [source run]" if continuation.get("prepare_on_start") else "Continuation from the completed [source run]"
            parts += [f"{label}(../{source}) at update "
                      f"{continuation['source_steps']} to **{continuation['target_steps']} total updates**, "
                      "preserving native optimizer, scheduler, RNG and buffered rollout state. "
                      "Warm-up is not repeated. Copied diagnostics retain their original measurements; "
                      "the source run and its full post-attack evaluation remain unchanged.", ""]
            if continuation.get("prepare_on_start"):
                parts += ["Each job waits for successful completion of its matching source job. "
                          "After extending the first attacker, it performs defender round 1, then attacker "
                          "and defender rounds 2 and 3. Those later attacker phases each restart the "
                          "50-update warm-up with a fresh optimizer while continuing the attacker weights. "
                          "Source copies are committed once per detector before training; allocation "
                          "resumption preserves the current phase's native training state.", ""]
        cancellation = path.parent / "cancellation.json"
        if cancellation.exists():
            parts += ["Cancelled at the user's request; saved artifacts are preserved.", ""]
        if manifest.get("metrics_file"):
            metrics = Path(manifest["metrics_file"]).relative_to(ROOT)
            parts += [f"[Shared metrics after every evaluation](../{metrics}). "
                      "Rows identify the race, round, phase, detector and cumulative attacker updates. "
                      "Concurrent writes are locked; replayed measurements do not add duplicate rows.", ""]
        for name, job in manifest["jobs"].items():
            directory = Path(job["directory"])
            receipt_path = directory / "receipt.json"
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
                "status": "cancelled_by_user" if cancellation.exists() else "submitted"}
            history_path = directory / "history.json"
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            parts += [f"### {name}: job {job['job_id']}", "",
                      f"Status: {receipt['status']}; completed rounds: {len(history)}/{protocol['rounds']}.", ""]
            if name in retained:
                parts += [f"Retained attacker: `{retained[name]['checkpoint']}`, "
                          f"trained for **{retained[name]['updates']} updates**. "
                          "Initial evaluations reuse the saved texts and scores after identity checks. "
                          "Post-defense evaluation reuses exactly those attack texts.", ""]
            if receipt.get("error"):
                parts += [receipt["error"], ""]
            prefix = f"race100-{path.parent.name}-{name}"
            layout = directory / "round_1/attacker/microbatch_profile.json"
            if layout.exists():
                value = json.loads(layout.read_text())
                parts += [f"Selected physical microbatch: **{value['selected_microbatch']}**; "
                          f"gradient accumulation: **{value['effective_completions'] // value['selected_microbatch']}**. "
                          "These probes perform zero optimizer updates and restore RNG state. "
                          "Timings cover reference forward and GRPO forward/backward; generation and reward "
                          "scoring are excluded. Training step-time curves include the real training path.", "",
                          table(value["measurements"], ["microbatch", "status", "median_seconds", "peak_allocated_gib"]), ""]
            if history:
                reuse = json.loads((directory / "reuse_decision.json").read_text())
                parts += [render_stage(ROOT, prefix, payload={"history": history, "reuse": reuse}), ""]
            rows = []
            for item in history:
                for phase in ("before", "after_attack", "after_defense"):
                    measurement = item[phase]
                    for record in measurement["records"]:
                        if record["detector"] != "ensemble_mean":
                            rows.append({"round": item["round"], "phase": phase, **record,
                                         "semantic_similarity": measurement.get("mean_semantic_similarity")})
            recorded = {(row["round"], row["phase"], row["detector"]) for row in rows}
            for journal in sorted(directory.glob("round_*/*/evaluation.json")):
                index = int(journal.parent.parent.name.removeprefix("round_"))
                phase = journal.parent.name
                measurement = json.loads(journal.read_text())
                for record in measurement["records"]:
                    if record["detector"] != "ensemble_mean" and (index, phase, record["detector"]) not in recorded:
                        rows.append({"round": index, "phase": phase, **record,
                                     "semantic_similarity": measurement.get("mean_semantic_similarity")})
            parts += [table(rows, ["round", "phase", "detector", "attack_success_rate", "ai_detection_rate",
                                  "human_false_positive_rate", "roc_auc", "semantic_similarity",
                                  "num_generated", "num_ineligible_pairs"]), ""]
            diagnostics = []
            diagnostic_series = []
            for round_dir in sorted(directory.glob("round_*")):
                index = int(round_dir.name.removeprefix("round_"))
                points = []
                for diagnostic in sorted((round_dir / "attacker/diagnostics").glob("step_*/metrics.json")):
                    value = json.loads(diagnostic.read_text())
                    step = int(diagnostic.parent.name.removeprefix("step_"))
                    for record in value["records"]:
                        if record["detector"] != "ensemble_mean":
                            row = {"round": index, "step": step, **record,
                                   "num_generated": value["num_generated"],
                                   "semantic_similarity": value.get("mean_semantic_similarity")}
                            diagnostics.append(row)
                            points.append(row)
                if points:
                    diagnostic_series.append((f"Round {index}", [row["step"] for row in points],
                                              [row["attack_success_rate"] for row in points]))
            if diagnostics:
                sample_counts = ", ".join(str(count) for count in sorted({row["num_generated"] for row in diagnostics}))
                parts += [f"Fixed diagnostic measurements ({sample_counts} prompts per measurement); logging only. Lines connect measured points without smoothing. "
                          "The 500-text post-attack evaluation is reported separately above.", "",
                          table(diagnostics, ["round", "step", "detector", "num_generated", "attack_success_rate",
                                              "ai_detection_rate", "human_false_positive_rate", "semantic_similarity"]), ""]
                figure = lines(ROOT / "docs/figures/research", prefix + "-diagnostic-evasion",
                               name + " evasion on fixed diagnostic prompts", "Attacker optimizer updates within round",
                               "Attack success rate", diagnostic_series, bounded=True)
                if figure is not None:
                    relative = figure.relative_to(report.parent)
                    parts += [f"![Diagnostic evasion over attacker updates]({relative})", "",
                              f"[Source CSV]({relative.with_suffix('.csv')}) · "
                              f"[Vector SVG]({relative.with_suffix('.svg')}).", ""]
            logs = []
            offset = 0
            for index in range(1, protocol["rounds"] + 1):
                training = directory / f"round_{index}/training_history.json"
                if not training.exists():
                    checkpoints = sorted((directory / f"round_{index}/attacker").glob("checkpoint-*/trainer_state.json"),
                                         key=lambda path: int(path.parent.name.split("-")[-1]))
                    if checkpoints:
                        phase_logs = json.loads(checkpoints[-1].read_text())["log_history"]
                        logs += [{**row, "race_update": offset + row["step"]}
                                 for row in phase_logs if "step" in row]
                        offset += max((row.get("step", 0) for row in phase_logs), default=0)
                if training.exists():
                    phase_logs = json.loads(training.read_text())
                    logs += [{**row, "race_update": offset + row["step"]}
                             for row in phase_logs if "step" in row]
                    offset += max((row.get("step", 0) for row in phase_logs), default=0)
            metrics = ["reward", "loss", "learning_rate"]
            if protocol.get("execution", {}).get("attacker_training", {}).get("grpo_beta", 0):
                metrics += ["kl", "step_time"]
            for metric in metrics:
                selected = [row for row in logs if metric in row]
                figure = lines(ROOT / "docs/figures/research", prefix + "-" + metric,
                               name + " attacker " + metric, "Cumulative attacker optimizer updates", metric,
                               [(name, [row["race_update"] for row in selected], [row[metric] for row in selected])])
                if figure is not None:
                    relative = figure.relative_to(report.parent)
                    parts += [f"![{figure.stem}]({relative})", "",
                              f"[Source CSV]({relative.with_suffix('.csv')}) · "
                              f"[Vector SVG]({relative.with_suffix('.svg')}).", ""]
    return parts


def short_probes(report):
    from scripts.plot_research import bars, lines
    from sklearn.metrics import roc_curve

    parts = []
    for path in sorted((OUTPUT / "defender_comparison/probes_10").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        parts += [f"## Ten-update probes: {path.parent.name}", "",
                  "Each Qwen attacker starts from base and receives ten optimizer updates against one "
                  "frozen warmed defender. Evaluation uses the existing 500 reserved prompts and fixed "
                  "human passages. This set has informed training-budget decisions and is therefore a "
                  "held-out diagnostic set, not an untouched final test. Evasion counts valid successful "
                  "outputs over all 500 generations; detection metrics use eligible pairs and rejection "
                  "counts remain visible.", "",
                  "The older reference is a 502-update attacker trained against all seven defenders "
                  "with an 11,988-update schedule. These probes use individual targets and ten-update "
                  "schedules. This is an unmatched descriptive comparison; it does not isolate update count.", "",
                  f"[Shared probe metrics CSV](../{Path(manifest['metrics_file']).relative_to(ROOT)}).", ""]
        if "split_audit" in manifest:
            audit = manifest["split_audit"]
            parts += [f"Split audit: {audit.get('scored_prompts')} evaluation rows, "
                      f"{audit.get('unique_scored_prompts', 'unrecorded')} distinct prompts; "
                      f"exact fitting-prompt overlap: {audit.get('exact_prompt_overlap_with_fitting', 'unrecorded')}. "
                      "The original repeated-prompt rows remain in the comparison.", ""]
        reference_path = Path(manifest["reference"])
        reference = json.loads(reference_path.read_text())
        baseline = json.loads((OUTPUT / "defender_comparison/warmup/measurements.json").read_text())
        rows, curves = [], []
        for name, job in manifest["jobs"].items():
            directory = Path(job["directory"])
            receipt_path = directory / "receipt.json"
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"status": "submitted"}
            parts += [f"### {name}: job {job['job_id']}, {receipt['status']}", ""]
            if receipt.get("error"):
                parts += [receipt["error"], ""]
            result_path = directory / "results.json"
            if not result_path.exists():
                continue
            value = json.loads(result_path.read_text())
            detector = manifest["defenders"][name]
            current = next(row for row in value["records"] if row["detector"] == detector)
            before = next(row for row in baseline["records"] if row["detector"] == detector)
            older = next(row for row in reference["records"] if row["detector"] == detector)
            for label, updates, record, semantic in [
                ("Base Qwen", 0, before, None),
                ("Individual target", 10, current, value["mean_semantic_similarity"]),
                ("Seven-defender ensemble target", 502, older, reference["mean_semantic_similarity"]),
            ]:
                rows.append({"detector": detector, "attacker_training": label, "updates": updates,
                             **record, "mean_semantic_similarity": semantic})
            series = []
            for label, score_path in [
                ("Base Qwen", OUTPUT / "defender_comparison/warmup/measurements.json"),
                ("10 individual updates", directory / "evaluation/scores/measurements.json"),
                ("502 ensemble updates", reference_path.parent / "scores/measurements.json"),
            ]:
                scores = json.loads(score_path.read_text())["scores"][detector]
                n = len(scores) // 2
                if n:
                    fpr, tpr, _ = roc_curve([1] * n + [0] * n, scores)
                    series.append((label, fpr.tolist(), tpr.tolist()))
            curves.append(lines(ROOT / "docs/figures/research", f"probe10-{path.parent.name}-{name}-roc",
                                name + ": diagnostic ROC comparison", "Human false-positive rate",
                                "AI detection rate", series, bounded=True))
        if rows:
            parts += [table(rows, ["detector", "attacker_training", "updates", "ai_detection_rate",
                                  "human_false_positive_rate", "attack_success_rate", "roc_auc",
                                  "mean_semantic_similarity", "num_generated", "num_ineligible_pairs"]), ""]
            curves.append(bars(ROOT / "docs/figures/research", f"probe10-{path.parent.name}-evasion",
                               "Descriptive evasion comparison: different attacker training targets",
                               "Attack success rate", [(label, {row["detector"]: row["attack_success_rate"]
                                                               for row in rows if row["attacker_training"] == label})
                                                       for label in ("Base Qwen", "Individual target",
                                                                     "Seven-defender ensemble target")]))
        for figure in curves:
            if figure is not None:
                relative = figure.relative_to(report.parent)
                parts += [f"![{figure.stem}]({relative})", "",
                          f"[Source CSV]({relative.with_suffix('.csv')}) · "
                          f"[Vector SVG]({relative.with_suffix('.svg')}).", ""]
    return parts


def synonym_robustness(report):
    from scripts.plot_research import bars, lines
    import csv

    parts = []
    for path in sorted((OUTPUT / "defender_comparison/synonym_robustness").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        receipt = path.parent / "receipt.json"
        state = json.loads(receipt.read_text()) if receipt.exists() else {"status": "held" if manifest["held"] else "submitted"}
        parts += [f"## Synonym sensitivity: {path.parent.name}", "",
                  f"Job `{manifest.get('job_id', 'unrecorded')}`: {state['status']}.", "",
                  "The same 500 saved base-Qwen outputs are edited without RL training or detector feedback. "
                  "A frozen Gemma 1.1 2B instruction model is asked to select context-appropriate WordNet candidates. "
                  "Edits affect at most 5% of eligible content words in each text; names, numbers, punctuation "
                  "and whitespace remain unchanged. Candidate words must be in their lemma form, which limits "
                  "coverage but avoids changing inflection. Every accepted and rejected proposal is recorded. "
                  "All rewrites are frozen before the seven warmed detectors score them.", "",
                  "Scores and detection decisions are paired with the original saved outputs at threshold 0.5. "
                  "Human passages and their saved scores remain fixed, so their false-positive rate is unchanged "
                  "by construction. This experiment does not test synonym edits to human writing. "
                  "The 500 diagnostic rows contain 488 distinct prompts. A drop measures lexical sensitivity "
                  "under this perturbation; it does not by itself establish training-set overfitting. "
                  "E5 similarity is a meaning proxy, not a guarantee of grammatical or factual preservation.", ""]
        if state.get("error"):
            parts += [state["error"], ""]
        result = path.parent / "results.json"
        if not result.exists():
            count = len(list((path.parent / "variants").glob("*.json")))
            parts += [f"Saved rewrite rows: {count}/500. No completed detector comparison is available yet.", ""]
            continue
        value = json.loads(result.read_text())
        parts += [f"**Initial 5% cap. Actual eligible words edited: "
                  f"{100 * value['actual_fraction_of_eligible_words']:.2f}% across {value['num_pairs']} texts.**", "",
                  f"Changed texts: {value['num_changed']}/{value['num_pairs']}; accepted edits: "
                  f"{value['num_edits']}; actual share of eligible content words changed: "
                  f"{100 * value['actual_fraction_of_eligible_words']:.2f}%. Mean original-versus-rewritten "
                  f"E5 cosine: {value['mean_semantic_original_vs_rewritten']:.4f}; changed-text mean: "
                  f"{value['mean_semantic_changed_only']:.4f}. Rejected proposals: {value['num_rejected_proposals']}.", "",
                  "Unchanged texts stay in the 500-row comparison; the changed-only score delta is also reported. "
                  "A negative score delta means less AI evidence after editing. Detection flips count both directions.", "",
                  table(value["records"], ["detector", "mean_score_before", "mean_score_after", "mean_score_delta",
                        "changed_only_mean_score_delta", "ai_detection_before", "ai_detection_after",
                        "detected_to_undetected", "undetected_to_detected", "roc_auc_before", "roc_auc_after"]), "",
                  f"[All paired scores](../{(path.parent / 'paired_scores.csv').relative_to(ROOT)}) · "
                  f"[Original texts, rewrites and edit logs](../{(path.parent / 'variants.json').relative_to(ROOT)}).", ""]
        paired = list(csv.DictReader((path.parent / "paired_scores.csv").open()))
        series = []
        for row in value["records"]:
            deltas = sorted(float(item["score_delta"]) for item in paired if item["detector"] == row["detector"])
            series.append((row["detector"], deltas, [(index + 1) / len(deltas) for index in range(len(deltas))]))
        figures = [bars(ROOT / "docs/figures/research", "synonyms-" + path.parent.name + "-detection",
                        "Frozen warmed detectors: original and synonym-edited base-Qwen text", "AI detection rate",
                        [(label, {row["detector"]: row[key] for row in value["records"]})
                         for label, key in [("Original", "ai_detection_before"), ("Synonyms", "ai_detection_after")]]),
                   lines(ROOT / "docs/figures/research", "synonyms-" + path.parent.name + "-score-deltas",
                         "Distribution of paired score changes", "Edited minus original AI score",
                         "Cumulative fraction of evaluation rows", series, bounded=True)]
        for figure in figures:
            if figure is not None:
                relative = figure.relative_to(report.parent)
                parts += [f"![{figure.stem}]({relative})", "",
                          f"[Source CSV]({relative.with_suffix('.csv')}) · "
                          f"[Vector SVG]({relative.with_suffix('.svg')}).", ""]
    return parts


def synonym_intensity(report):
    from scripts.plot_research import lines

    parts = []
    for path in sorted((OUTPUT / "defender_comparison/synonym_intensity").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        parts += [f"## Reviewed synonym intensity: {path.parent.name}", "",
                  "The first 100 saved base-Qwen evaluation rows receive a 25% edit cap, measured over "
                  "eligible content words. Each variant starts from the original. Names, numbers, punctuation and whitespace "
                  "remain fixed. There is no detector feedback or training. Only frozen warmed RoBERTa "
                  "and Gemma are evaluated, using their original thresholds and matching fixed human passages.", "",
                  "Preparation and scoring use separate checkpointed development allocations. Every proposed "
                  "substitution and every 25% rewrite is reviewed before new detector scoring. Unsafe edits "
                  "are removed or replaced by an allowed contextual synonym at the same proposed position, "
                  "even when the actual rate falls below the target. Complete edit objects in truncated "
                  "editor JSON are recovered before review; incomplete objects are ignored and raw responses "
                  "are retained. Coherence means preserving "
                  "the original text's sense and grammar; existing base-model factual or question-answer "
                  "artifacts are retained. This is an agent review, not independent human validation. "
                  "The review and variant hashes are frozen before releasing the scoring job.", ""]
        for stage, job in manifest["jobs"].items():
            receipt = path.parent / f"{stage}_receipt.json"
            state = json.loads(receipt.read_text()) if receipt.exists() else {"status": "held" if (manifest["held"] if stage == "prepare" else manifest["score_held"]) else "submitted"}
            parts += [f"{stage}: job `{job}`, {state['status']}.", ""]
            if state.get("error"):
                parts += [state["error"], ""]
        result = path.parent / "results.json"
        gallery = path.parent / "coherence_review.md"
        if gallery.exists():
            parts += [f"[Read the 100 originals, initial 5%-cap edits and reviewed 25%-cap edits](../{gallery.relative_to(ROOT)}).", ""]
        if not result.exists():
            if (path.parent / "review_receipt.json").exists():
                reviewed = []
                for condition, cap in manifest["conditions"].items():
                    if condition != "p25":
                        continue
                    variants = json.loads((path.parent / condition / "reviewed_variants.json").read_text())
                    edits = sum(len(row["edits"]) for row in variants)
                    eligible = sum(row["eligible_content_words"] for row in variants)
                    reviewed.append({"condition": condition, "nominal_cap": cap,
                                     "actual_edit_fraction": edits / eligible,
                                     "retained_edits": edits,
                                     "changed_texts": sum(bool(row["edits"]) for row in variants)})
                parts += [table(reviewed, ["condition", "nominal_cap", "actual_edit_fraction",
                          "retained_edits", "changed_texts"]), "",
                          "Coherence review is complete and frozen. Detector measurements are still pending; "
                          "these edit counts are not evidence of detector robustness.", ""]
            for condition in manifest["conditions"]:
                if condition != "p25":
                    continue
                count = len(list((path.parent / condition / "variants").glob("*.json")))
                parts += [f"{condition}: {count}/100 saved proposal rows.", ""]
            parts += ["No completed intensity comparison is available yet.", ""]
            continue
        value = json.loads(result.read_text())
        conditions = [row for row in value["conditions"] if row["condition"] == "p25"]
        for condition in conditions:
            parts += [f"**25% cap. Actual eligible words edited: "
                      f"{100 * condition['actual_fraction_of_eligible_words']:.2f}% across {condition['num_pairs']} texts.**", ""]
        parts += [table(conditions, ["condition", "target_rate", "actual_fraction_of_eligible_words",
                  "num_proposed_edits", "num_review_corrections", "num_coherence_rejections",
                  "num_edits", "num_changed"]), "",
                  "Rejected proposals include positions corrected to a safer synonym. The final edit count "
                  "includes those corrections. Unchanged rows remain in the 100-row comparison.", ""]
        rows = []
        for condition in conditions:
            for record in condition["records"]:
                rows.append({"target_edit_cap": condition["target_rate"],
                             "actual_edit_fraction": condition["actual_fraction_of_eligible_words"],
                             "num_changed": condition["num_changed"], "num_edits": condition["num_edits"],
                             **record})
        parts += [table(rows, ["detector", "target_edit_cap", "actual_edit_fraction", "num_changed", "num_edits",
                              "ai_detection_before", "ai_detection_after", "mean_score_before", "mean_score_after",
                              "mean_score_delta", "detected_to_undetected", "human_false_positive_rate",
                              "roc_auc_before", "roc_auc_after"]), "",
                  f"[Frozen review decisions](../{(path.parent / 'review.json').relative_to(ROOT)}). "
                  "Rates are observed edit fractions, not assumed achievement of the nominal caps. "
                  "The baseline uses these 100 rows, not the earlier 500-row aggregate. Curves use actual "
                  "edit fractions on the horizontal axis. Each AI miss is one percentage point on this "
                  "100-text set. Human false positives use unchanged human controls, so they are shared "
                  "across edit conditions. Connecting lines "
                  "are visual guides between the unedited baseline and the reviewed condition.", ""]
        for key, baseline_key, title, ylabel in [
            ("ai_detection_after", "ai_detection_before", "Detection after reviewed synonym edits", "AI detection rate"),
            ("mean_score_after", "mean_score_before", "AI scores after reviewed synonym edits", "Mean AI score"),
        ]:
            series = []
            for detector in manifest["detector_names"]:
                selected = sorted([row for row in rows if row["detector"] == detector], key=lambda row: row["target_edit_cap"])
                label = {"roberta": "RoBERTa", "decoder_classifier": "Gemma"}[detector]
                series.append((label, [0.0] + [row["actual_edit_fraction"] for row in selected],
                               [selected[0][baseline_key]] + [row[key] for row in selected]))
            figure = lines(ROOT / "docs/figures/research", f"synonym-intensity-{path.parent.name}-{key}",
                           title, "Actual edited fraction of eligible content words", ylabel, series,
                           bounded=key == "ai_detection_after")
            if figure is not None:
                relative = figure.relative_to(report.parent)
                parts += [f"![{figure.stem}]({relative})", "",
                          f"[Source CSV]({relative.with_suffix('.csv')}) · "
                          f"[Vector SVG]({relative.with_suffix('.svg')}).", ""]
    return parts


def _update_report():
    selection = OUTPUT / "report_selection.json"
    if selection.exists():
        from scripts.report_retained_research import update_retained_report
        return update_retained_report(ROOT, OUTPUT, json.loads(selection.read_text()))
    report = ROOT / "docs/research_results.md"
    prefix = report.read_text().split(MARKER)[0].rstrip()
    parts = [MARKER, "", "## Measurements from completed stages", ""]
    any_completed = False
    submissions = OUTPUT / "submissions.jsonl"
    latest = {item["stage"]: item for item in map(json.loads, submissions.read_text().splitlines())} if submissions.exists() else {}
    for receipt_path in sorted(OUTPUT.glob("*.json")):
        receipt = json.loads(receipt_path.read_text())
        if "stage" not in receipt:
            continue
        label = {"text_notebook": "Text detection notebook", "image_notebook": "Image detection notebook",
                 "grpo_text": "GRPO", "multispin": "SPIN", "ddpo_image": "DDPO",
                 "benchmark": "Text benchmarks", "arms_race": "Attacker and defender", "meta_adapt": "Full MAML",
                 "comparison_baseline": "Base-Qwen detection", "comparison_warmup": "Defender warm-up",
                 "comparison_rl_encoder": "RL: ModernBERT-large", "comparison_rl_decoder": "RL: decoder",
                 "comparison_rl_ensemble": "RL: ensemble", "comparison_race_encoder": "Race: ModernBERT-large",
                 "comparison_race_decoder": "Race: decoder", "comparison_race_ensemble": "Race: ensemble"}.get(
                     receipt["stage"], receipt["stage"])
        job_id = receipt["slurm_job_id"]
        status = receipt.get("status", "running")
        if receipt["stage"] in latest and latest[receipt["stage"]]["job_id"] != job_id:
            job_id = latest[receipt["stage"]]["job_id"]
            status = "submitted; awaiting execution receipt"
        lines = []
        in_status_table = False
        for line in prefix.splitlines():
            if line == "| Stage | Protocol | Current state |":
                in_status_table = True
            elif not line.startswith("|"):
                in_status_table = False
            if in_status_table and line.startswith(f"| {label} |"):
                line = line.rsplit("|", 2)[0] + f"| Job {job_id}: {status} |"
            lines.append(line)
        prefix = "\n".join(lines)
        if status != "completed":
            if status in {"failed", "paused"}:
                error = (receipt.get("error", "").strip() or "See the execution receipt.").splitlines()[-1]
                if len(error) > 600:
                    error = "... " + error[-600:]
                parts += [f"### {receipt['stage']}: {status}", "",
                          f"Job `{job_id}`. {error}", "",
                          f"[Execution receipt](../{receipt_path.relative_to(ROOT)}). "
                          "No completed-stage performance is attributed to this attempt.", ""]
            continue
        any_completed = True
        stage = receipt["stage"]
        parts += [f"### {stage}", "", f"Job `{receipt['slurm_job_id']}`; source fingerprint "
                  f"`{receipt['source_fingerprint']}`.", ""]
        if stage.startswith("comparison_"):
            from scripts.report_defender_comparison import render_stage
            parts += [render_stage(ROOT, stage), ""]
        elif stage in ("text_notebook", "image_notebook"):
            payload = json.loads((OUTPUT / f"{stage}_measurements.json").read_text())
            parts += [notebook_analysis(payload, level=4), ""]
            selection_path = OUTPUT / "defender_comparison/notebook_selection.json"
            if stage == "text_notebook" and selection_path.exists():
                selection = json.loads(selection_path.read_text())
                rows = [{"family": role, "model": name,
                         selection["metric"]: selection["selected_scores"][role]}
                        for role, name in selection["selected"].items()]
                parts += ["#### Notebook defenders selected for the Qwen comparison", "",
                          table(rows, ["family", "model", selection["metric"]]), "",
                          "Selection uses MAGE test macro F1, excludes Qwen decoder classifiers, "
                          "and retains ModernBERT-large. These are selection scores; the independent "
                          "Qwen baseline and warm-up evaluations are still separate experiments.", ""]
        else:
            paths = {
                "grpo_text": [ROOT / "results/text_evasion/grpo/evaluation.json"],
                "ddpo_image": [ROOT / "results/image_evasion/ddpo/evaluation.json"],
                "benchmark": [OUTPUT / "benchmark" / name / "benchmark_results.json" for name in ("hc3", "cnn_dailymail")],
                "arms_race": [ROOT / "results/arms_race/text/arms_race_history.json"],
                "meta_adapt": [ROOT / "results/meta_adapt/adaptation_comparison.json"],
            }.get(stage, [])
            if stage == "multispin":
                parts += ["Training completed. This stage reports feature diagnostics; the subsequent "
                          "generation benchmark supplies its evasion measurements.", ""]
            for path in paths:
                value = json.loads(path.read_text())
                parts += [f"Artifact: [{path.name}](../{path.relative_to(ROOT)}).", ""]
                if stage == "benchmark":
                    rows = [{"method": name, **metrics} for name, metrics in value["results"].items()]
                    parts += [table(rows, ["method", "status", "num_succeeded", "mean_evasion",
                                          "mean_semantic_similarity", "attack_success_rate"]), ""]
                elif isinstance(value, dict) and all(not isinstance(item, (dict, list)) for item in value.values()):
                    parts += [table([{"metric": name, "value": item} for name, item in value.items()],
                                    ["metric", "value"]), ""]
                elif stage == "arms_race":
                    rows = [{"round": row["round"],
                             "ASR before attack": row["before_attacker_update"]["attack_success_rate"],
                             "ASR after attack": row["after_attacker_update"]["attack_success_rate"],
                             "ASR after defense": row["after_defender_update"]["attack_success_rate"],
                             **row["update_effects"]} for row in value]
                    parts += [table(rows, list(rows[0])), "",
                              "A positive attacker success change followed by defender recovery describes "
                              "a paired round. Ten rounds do not establish an equilibrium.", ""]
                elif stage == "meta_adapt":
                    availability_path = path.with_name("detector_availability.json")
                    if availability_path.exists():
                        availability = json.loads(availability_path.read_text())
                        parts += ["Detector admission checks:", "", table(
                            [{"detector": name, **details} for name, details in availability.items()],
                            ["detector", "status", "reason", "checkpoint_id"]), ""]
                    manifest = json.loads(path.with_name("prompt_manifest.json").read_text())
                    speedup = "undefined (null)" if value["speedup"] is None else f"{value['speedup']:.4f}"
                    curves = {key: dict(zip(value[key]["steps"], value[key]["evasion"])) for key in ("meta", "pre_meta")}
                    rows = [{"step": step, "meta evasion": curves["meta"].get(step),
                             "pre-meta evasion": curves["pre_meta"].get(step)}
                            for step in sorted(set(curves["meta"]) | set(curves["pre_meta"]))]
                    parts += [table(rows, ["step", "meta evasion", "pre-meta evasion"]), "",
                              f"Measured adaptation speedup: {speedup}. A null value means that "
                              "the threshold-crossing ratio is undefined, including a curve that never "
                              "crosses 0.7 or a meta initialization already above it at step zero. "
                              "This is an update-count ratio, not a wall-clock speedup.", "",
                              f"Reserved detector: `{manifest['held_out_detector']}`. Meta-training detectors: "
                              f"{', '.join(manifest['training_detectors'])}. Prompt partitions contain "
                              f"{len(manifest['meta_training'])} meta-training, {len(manifest['adaptation'])} "
                              f"adaptation and {len(manifest['evaluation'])} reserved evaluation prompts. "
                              "Each curve measurement scores the first 20 reserved evaluation prompts.", ""]
                else:
                    parts += ["```json", json.dumps(value, indent=2), "```", ""]
        folder = {"grpo_text": "text_evasion/grpo", "multispin": "text_evasion/multispin",
                  "ddpo_image": "image_evasion/ddpo", "meta_adapt": "meta_adapt"}.get(stage)
        if folder:
            training_path = ROOT / "results" / folder / "training_history.json"
            if training_path.exists():
                rows = json.loads(training_path.read_text())
                columns = [key for key in ("step", "epoch", "iteration", "global_step", "outer_step", "loss",
                                           "spin_loss", "meta_loss", "reward", "mean_reward", "ppo_loss", "approx_kl")
                           if any(key in row for row in rows)]
                shown = rows if len(rows) <= 20 else rows[:5] + rows[-5:]
                parts += [f"Training history: [{len(rows)} recorded rows](../{training_path.relative_to(ROOT)}). "
                          "Tables with more than 20 rows show the first and last five; plots use all recorded points.", ""]
                if shown and columns:
                    parts += [table(shown, columns), ""]
        from scripts.plot_research import stage_figures
        for figure, caption in stage_figures(ROOT, stage, payload if stage in ("text_notebook", "image_notebook") else None):
            relative = figure.relative_to(report.parent)
            parts += [f"![{figure.stem}]({relative})", "", caption + " "
                      f"[Source CSV]({relative.with_suffix('.csv')}) · "
                      f"[Vector SVG]({relative.with_suffix('.svg')}).", ""]
    parts += checkpoint_evaluations(report)
    parts += architecture_races(report)
    parts += short_probes(report)
    parts += synonym_robustness(report)
    parts += synonym_intensity(report)
    if not any_completed:
        parts += ["No complete stage measurements are available yet. Interrupted attempts remain in the execution receipts.", ""]
    if any_completed:
        prefix = prefix.replace("The full research runs have not produced new performance measurements yet.",
                                "Completed-stage measurements are recorded below; unfinished stages remain identified.")
        prefix = prefix.replace("No new performance table is available yet.", "Completed-stage tables appear below.")
    import os
    temporary = report.with_name(f".{report.name}.{os.getpid()}.tmp")
    temporary.write_text(prefix + "\n\n" + "\n".join(parts))
    temporary.replace(report)


if __name__ == "__main__":
    update_report()
