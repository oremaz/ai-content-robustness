import json

import nbformat
import pytest

from scripts import report_research


def fixture(tmp_path, monkeypatch):
    output = tmp_path / "results/research_workflow"
    output.mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/research_results.md").write_text(
        "# Evidence\n\n| Stage | Protocol | Current state |\n|---|---|---|\n"
        "| Text detection notebook | Full protocol | pending |\n"
    )
    monkeypatch.setattr(report_research, "ROOT", tmp_path)
    monkeypatch.setattr(report_research, "OUTPUT", output)
    payload = {"job_id": "123", "tables": {"MAGE test": [
        {"model": "fixture", "accuracy": 0.8, "f1_macro": 0.79, "roc_auc": 0.85},
    ]}}
    (output / "text_notebook_measurements.json").write_text(json.dumps(payload))
    return output, payload


def test_report_includes_only_completed_stages_and_updates_without_duplication(tmp_path, monkeypatch):
    output, payload = fixture(tmp_path, monkeypatch)
    receipt = {"stage": "text_notebook", "status": "failed", "slurm_job_id": "123", "source_fingerprint": "abc"}
    (output / "text_notebook.json").write_text(json.dumps(receipt))
    report_research.update_report()
    path = tmp_path / "docs/research_results.md"
    assert "0.7900" not in path.read_text()
    assert "Job 123: failed" in path.read_text()
    assert not (tmp_path / "docs/figures").exists()
    receipt["status"] = "completed"
    (output / "text_notebook.json").write_text(json.dumps(receipt))
    report_research.update_report()
    text = path.read_text()
    assert "0.7900" in text and "source fingerprint `abc`" in text
    assert "figures/research/text_notebook-transfer.png" in text
    assert (tmp_path / "docs/figures/research/text_notebook-transfer.svg").exists()
    report_research.update_report()
    assert path.read_text() == text


def test_status_updates_do_not_overwrite_a_separate_dependency_table(tmp_path, monkeypatch):
    output, _ = fixture(tmp_path, monkeypatch)
    path = tmp_path / "docs/research_results.md"
    path.write_text(path.read_text() + "\n| Stage | Job | Prerequisites |\n|---|---|---|\n"
                    "| Text detection notebook | 123 | afterok:42 |\n")
    (output / "text_notebook.json").write_text(json.dumps({
        "stage": "text_notebook", "status": "failed", "slurm_job_id": "123"}))
    report_research.update_report()
    assert "| Text detection notebook | 123 | afterok:42 |" in path.read_text()
    assert "| Text detection notebook | Full protocol | Job 123: failed |" in path.read_text()


def test_notebook_interpretation_and_outputs_update_after_completed_execution(tmp_path, monkeypatch):
    output, payload = fixture(tmp_path, monkeypatch)
    directory = tmp_path / "ai_content_detector/notebooks"
    directory.mkdir(parents=True)
    source = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("original dataset loader"),
        nbformat.v4.new_markdown_cell("## Analysis and Lessons Learned\nHistorical interpretation"),
    ])
    source_path = directory / "bench-aitextdetect.ipynb"
    nbformat.write(source, source_path)
    executed = nbformat.from_dict(source)
    executed.cells[0].source = "equivalent offline dataset loader"
    executed.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="measured output")]
    executed_path = output / "executed.ipynb"
    nbformat.write(executed, executed_path)
    report_research.finalize_notebook("text_notebook", executed_path)
    saved = nbformat.read(source_path, as_version=4)
    assert saved.cells[0].source == "original dataset loader"
    assert saved.cells[0].outputs[0].text == "measured output"
    assert "0.7900" in saved.cells[1].source
    assert "Historical interpretation" not in saved.cells[1].source


def test_failed_notebook_report_preserves_cause_without_dumping_traceback(tmp_path, monkeypatch):
    output, _ = fixture(tmp_path, monkeypatch)
    receipt = {"stage": "text_notebook", "status": "failed", "slurm_job_id": "123",
               "error": "PapermillExecutionError: Traceback\n" + "model layer dump\n" * 3000
                        + ") got multiple values for keyword argument 'output_hidden_states'"}
    (output / "text_notebook.json").write_text(json.dumps(receipt))
    report_research.update_report()
    text = (tmp_path / "docs/research_results.md").read_text()
    assert "multiple values for keyword argument 'output_hidden_states'" in text
    assert "model layer dump" not in text
    assert "../results/research_workflow/text_notebook.json" in text
    assert "0.7900" not in text


def test_image_intervention_reports_actual_gate_and_undefined_metrics():
    payload = {"job_id": "123", "tables": {"external": [
        {"model": "fixture", "accuracy": 0.5, "f1_macro": 0.3333, "roc_auc": None},
    ]}, "intervention": [{"dataset": "external", "variant": "shortcut-resistant", "f1_delta": -0.1}],
               "passes_exploratory_gate": False}
    text = report_research.notebook_analysis(payload)
    assert "gate failed" in text
    assert "undefined" in text
    assert "0.3333" in text


@pytest.mark.parametrize("stage", ["text_notebook", "image_notebook"])
def test_workflow_exports_live_tables_and_updates_report_after_notebook(tmp_path, monkeypatch, stage):
    import pandas as pd
    from scripts import research_workflow, defender_selection

    output, payload = fixture(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(research_workflow, "ROOT", tmp_path)
    monkeypatch.setattr(research_workflow, "OUTPUT", output)
    monkeypatch.setattr(research_workflow, "source_provenance", lambda: {"source_fingerprint": "fixture"})
    monkeypatch.setattr(research_workflow.subprocess, "check_output", lambda *args, **kwargs: "revision")
    monkeypatch.setattr(research_workflow, "validate_prerequisites", lambda stage: None)
    selection_calls = []
    monkeypatch.setattr(defender_selection, "save_notebook_selection", lambda *args: selection_calls.append(args))
    directory = tmp_path / "ai_content_detector/notebooks"
    directory.mkdir(parents=True)
    nbformat.write(nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("## Analysis and Lessons Learned\nOld results"),
    ]), directory / research_workflow.NOTEBOOKS[stage])
    frame = pd.DataFrame(payload["tables"]["MAGE test"])

    def execute(input_path, output_path):
        if stage == "text_notebook":
            assert "Job 123: running" in (tmp_path / "docs/research_results.md").read_text()
        notebook = nbformat.read(input_path, as_version=4)
        context = {name: frame for name in ["df_all_results", "df_pile_results", "df_external_ood_results",
                                          "artifact_test_metrics", "cifake_test_metrics", "shoes_metrics"]}
        context.update(baseline_names=["fixture"], ALL_QLORA_MODELS=[], SELECTED_MODELS=["fixture"],
                       robust_comparison=pd.DataFrame([{"dataset": "external", "variant": "shortcut-resistant",
                                                        "accuracy": 0.5, "f1_macro": 0.3, "f1_delta": -0.1}]),
                       passes_exploratory_gate=False)
        from sklearn.feature_extraction.text import TfidfVectorizer
        context["tfidf"] = TfidfVectorizer().fit(["human text", "generated text"])
        exec(notebook.cells[-1].source, context)
        nbformat.write(notebook, output_path)

    monkeypatch.setattr(research_workflow, "execute_notebook", execute)
    research_workflow.run_stage(stage)
    receipt = json.loads((output / f"{stage}.json").read_text())
    assert receipt["status"] == "completed"
    measurements = json.loads((output / f"{stage}_measurements.json").read_text())
    assert measurements["job_id"] == "123"
    assert len(measurements["tables"]) == 3
    assert "0.7900" in (tmp_path / "docs/research_results.md").read_text()
    assert len(selection_calls) == int(stage == "text_notebook")


def test_meta_report_preserves_undefined_speedup_and_detector_partition(tmp_path, monkeypatch):
    output, payload = fixture(tmp_path, monkeypatch)
    path = tmp_path / "results/meta_adapt"
    path.mkdir()
    curves = {"steps": [0, 5], "evasion": [0.2, 0.6]}
    (path / "adaptation_comparison.json").write_text(json.dumps({
        "meta": curves, "pre_meta": curves, "speedup": None,
    }))
    (path / "prompt_manifest.json").write_text(json.dumps({
        "held_out_detector": "heldout", "training_detectors": ["a", "b"],
        "meta_training": ["training"] * 500, "adaptation": ["adapt"] * 100,
        "evaluation": ["eval"] * 100,
    }))
    (path / "detector_availability.json").write_text(json.dumps({
        "heldout": {"status": "ready"},
        "style_embedding": {"status": "unavailable", "reason": "Support centroids missing"},
    }))
    (output / "meta_adapt.json").write_text(json.dumps({
        "stage": "meta_adapt", "status": "completed", "slurm_job_id": "456", "source_fingerprint": "abc",
    }))
    report_research.update_report()
    text = (tmp_path / "docs/research_results.md").read_text()
    assert "Reserved detector: `heldout`" in text
    assert "0.6000" in text and "speedup: undefined (null)" in text
    assert "first 20 reserved evaluation prompts" in text
    assert "Support centroids missing" in text
    assert "figures/research/maml-adaptation.png" in text


def test_race_report_includes_exact_paired_curves_and_tables(tmp_path, monkeypatch):
    import csv
    output, _ = fixture(tmp_path, monkeypatch)
    directory = tmp_path / "results/arms_race/text"
    directory.mkdir(parents=True)
    history = [{"round": index + 1,
                "before_attacker_update": {"attack_success_rate": before},
                "after_attacker_update": {"attack_success_rate": attack},
                "after_defender_update": {"attack_success_rate": defense},
                "update_effects": {"attacker_success_change": attack - before}}
               for index, (before, attack, defense) in enumerate([(0.2, 0.6, 0.3), (0.3, 0.7, 0.4)])]
    (directory / "arms_race_history.json").write_text(json.dumps(history))
    (output / "arms_race.json").write_text(json.dumps({
        "stage": "arms_race", "status": "completed", "slurm_job_id": "789", "source_fingerprint": "abc",
    }))
    report_research.update_report()
    text = (tmp_path / "docs/research_results.md").read_text()
    assert "ASR after defense" in text and "0.7000" in text
    assert "figures/research/arms-race-paired.png" in text
    directory = tmp_path / "docs/figures/research"
    assert (directory / "arms-race-paired.png").read_bytes().startswith(b"\x89PNG")
    with (directory / "arms-race-paired.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["y"]) for row in rows if row["series"] == "After defender update"] == [0.3, 0.4]
    assert [int(row["x"]) for row in rows if row["series"] == "Before attacker update"] == [1, 2]


def test_plots_keep_missing_measurements_missing(tmp_path):
    import csv
    from scripts.plot_research import lines
    figure = lines(tmp_path, "missing", "Test fixture", "Update", "Score",
                   [("Recorded", [0, 5, 10], [0.2, None, 0.6])], bounded=True)
    with figure.with_suffix(".csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["y"] for row in rows] == ["0.2", "", "0.6"]
