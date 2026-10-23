import json

import pytest

from scripts import research_chain


def fixture(tmp_path, monkeypatch):
    output = tmp_path / "results/research_workflow"
    output.mkdir(parents=True)
    monkeypatch.setattr(research_chain, "OUTPUT", output)
    monkeypatch.setattr(research_chain, "ROOT", tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    stage = "comparison_baseline"
    path = output / "chains/fixture/manifest.json"
    research_chain.write_record(path, {"jobs": {stage: {"job_id": "123"}}, "max_restarts": 2})
    return output, path, stage


def test_dependency_graph_never_exceeds_three_gpus_and_preserves_scientific_order():
    graph = {"image_active": []}
    graph.update({name: ["image_active" if target == "42" else target for _, target in dependencies]
                  for name, dependencies in research_chain.branches(42)})
    nodes = list(graph)
    for mask in range(1 << len(nodes)):
        completed = {name for index, name in enumerate(nodes) if mask & (1 << index)}
        if any(not set(graph[name]) <= completed for name in completed):
            continue
        available = {name for name in nodes if name not in completed and set(graph[name]) <= completed}
        assert len(available) <= 3, (completed, available)
    assert graph["comparison_warmup"] == ["comparison_baseline"]
    for group in ("encoder", "decoder", "ensemble"):
        assert "comparison_warmup" in graph[f"comparison_rl_{group}"]
        assert graph[f"comparison_race_{group}"] == [f"comparison_rl_{group}"]
    assert "image_notebook" not in graph["comparison_baseline"]


def test_native_chain_submits_held_regular_jobs_with_resolved_dependencies(tmp_path, monkeypatch):
    output, _, _ = fixture(tmp_path, monkeypatch)
    research_chain.write_record(output / "text_notebook.json", {"status": "completed"})
    calls = []

    def sbatch(command, **kwargs):
        calls.append(command)
        return str(100 + len(calls)) + ";jean-zay\n"

    monkeypatch.setattr(research_chain.subprocess, "check_output", sbatch)
    manifest = json.loads(research_chain.submit(42).read_text())
    assert len(calls) == 9
    assert all("--hold" in command and "--qos=qos_gpu_h100-t3" in command for command in calls)
    assert "--time=04:00:00" in calls[1] and "--cpus-per-task=32" in calls[1]
    assert "--time=04:00:00" in calls[2] and "--cpus-per-task=96" in calls[2]
    assert "--time=20:00:00" in calls[5] and "--cpus-per-task=48" in calls[5]
    assert "--cpus-per-task=96" in calls[0]
    assert "--dependency=afterany:42" in calls[0]
    assert not any(option.startswith("--dependency=") for option in calls[1])
    assert "--dependency=afterok:102" in calls[2]
    assert "--dependency=afterok:103,afterany:101" in calls[5]
    assert manifest["gpu_limit"] == 3


def test_submission_error_cancels_only_new_chain_jobs(tmp_path, monkeypatch):
    output, _, _ = fixture(tmp_path, monkeypatch)
    research_chain.write_record(output / "text_notebook.json", {"status": "completed"})
    responses = iter(["101", "102", "failure"])
    monkeypatch.setattr(research_chain.subprocess, "check_output", lambda *args, **kwargs: next(responses))
    cancelled = []
    monkeypatch.setattr(research_chain.subprocess, "run", lambda command, **kwargs: cancelled.append(command))
    with pytest.raises(RuntimeError, match="Unexpected sbatch"):
        research_chain.submit(42)
    assert cancelled == [["scancel", "101", "102"]]


def test_completed_stage_is_skipped_without_overwriting_receipt(tmp_path, monkeypatch):
    output, path, stage = fixture(tmp_path, monkeypatch)
    receipt = {"status": "completed", "slurm_job_id": "42"}
    research_chain.write_record(output / f"{stage}.json", receipt)
    research_chain.run(path, stage)
    assert json.loads((output / f"{stage}.json").read_text()) == receipt
    assert not (output / "submissions.jsonl").exists()


def test_saved_pause_requeues_same_job_and_preserves_stage_checkpoint(tmp_path, monkeypatch):
    from scripts import research_workflow
    output, path, stage = fixture(tmp_path, monkeypatch)
    checkpoint = output / "optimizer-state.pt"
    checkpoint.write_bytes(b"committed training state")

    def pause(name):
        research_chain.write_record(output / f"{name}.json", {"status": "paused", "slurm_job_id": "123"})
        raise SystemExit(75)

    monkeypatch.setattr(research_workflow, "run_stage", pause)
    commands = []
    monkeypatch.setattr(research_chain.subprocess, "run", lambda command, **kwargs: commands.append(command))
    research_chain.run(path, stage)
    assert commands == [["scontrol", "requeue", "123"]]
    assert checkpoint.read_bytes() == b"committed training state"
    assert (path.parent / "123-restart-0.json").exists()


@pytest.mark.parametrize("status", ["failed", "missing_receipt", "wrong_job"])
def test_scientific_failure_or_uncommitted_pause_never_requeues(tmp_path, monkeypatch, status):
    from scripts import research_workflow
    output, path, stage = fixture(tmp_path, monkeypatch)

    def fail(name):
        if status == "failed":
            raise ValueError("scientific failure")
        if status == "wrong_job":
            research_chain.write_record(output / f"{name}.json", {"status": "paused", "slurm_job_id": "42"})
        raise SystemExit(75)

    monkeypatch.setattr(research_workflow, "run_stage", fail)
    def unexpected(*args, **kwargs):
        pytest.fail("An invalid pause must not trigger requeue")
    monkeypatch.setattr(research_chain.subprocess, "run", unexpected)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        research_chain.run(path, stage)


def test_success_needs_the_current_job_completion_receipt(tmp_path, monkeypatch):
    from scripts import research_workflow
    output, path, stage = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(research_workflow, "run_stage", lambda name: research_chain.write_record(
        output / f"{name}.json", {"status": "completed", "slurm_job_id": "123"}))
    research_chain.run(path, stage)
    assert json.loads((output / f"{stage}.json").read_text())["status"] == "completed"
