import json

import pytest

from scripts import profile_baseline, research_workflow


@pytest.mark.parametrize("outcome", ["paused", "completed", "failed"])
def test_profile_handoff_preserves_progress_and_only_releases_safe_continuation(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(profile_baseline, "OUTPUT", tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "321")
    snapshots = iter([[], [{"completed": 8, "requested": 1000, "path": "saved.pt"}]])
    monkeypatch.setattr(profile_baseline, "generation_progress", lambda root: next(snapshots))
    timers = []

    class Timer:
        def __init__(self, seconds, callback):
            timers.append(self)
            self.cancelled = False

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(profile_baseline.threading, "Timer", Timer)

    def execute(stage):
        (tmp_path / "comparison_baseline.json").write_text(json.dumps({"status": outcome, "slurm_job_id": "321"}))
        if outcome == "paused":
            raise SystemExit(75)
        if outcome == "failed":
            raise ValueError("scientific failure")

    monkeypatch.setattr(research_workflow, "run_stage", execute)
    commands = []
    monkeypatch.setattr(profile_baseline.subprocess, "run", lambda command, **kwargs: commands.append(command))
    if outcome == "failed":
        with pytest.raises(ValueError, match="scientific failure"):
            profile_baseline.run(42, seconds=10)
        assert commands == []
    else:
        profile_baseline.run(42, seconds=10)
        assert commands[-1] == ["scontrol", "release", "42"]
        assert "Dependency=afterany:321" in commands[0]
        assert f"QOS={'qos_gpu_h100-dev' if outcome == 'completed' else 'qos_gpu_h100-t3'}" in commands[0]
    record = json.loads((tmp_path / "baseline_profile.json").read_text())
    assert record["new_completions"] == 8 and record["status"] == outcome
    assert timers[0].cancelled
