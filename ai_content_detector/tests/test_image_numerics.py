from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_nonfinite_loss_keeps_last_valid_optimizer_checkpoint_and_can_resume(tmp_path, monkeypatch):
    from ml_pipeline.tests.test_training_restart import make
    from ml_pipeline.utils import training_checkpoint
    monkeypatch.setattr(training_checkpoint.CheckpointClock, "due", lambda self: True)
    X = np.random.default_rng(13).normal(size=(48, 3)).astype(np.float32)
    y = np.arange(48) % 2
    baseline = make(tmp_path / "continuous")
    baseline.fit(X, y)
    interrupted = make(tmp_path / "restart")
    forward = interrupted.model.forward

    def invalid_on_third_batch(inputs):
        logits = forward(inputs)
        return logits * float("nan") if interrupted.model.calls == 3 else logits

    interrupted.model.forward = invalid_on_third_batch
    with pytest.raises(FloatingPointError, match="last safe restart checkpoint retained"):
        interrupted.fit(X, y)
    state = training_checkpoint.load_checkpoint(tmp_path / "restart/restart.pt")
    assert state["batch"] == 2 and np.isfinite(state["train_loss"])
    assert all(torch.isfinite(value).all() for value in state["model"].values())
    resumed = make(tmp_path / "restart")
    resumed.fit(X, y)
    for name, weight in baseline.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[name], weight), name
    assert resumed.history == baseline.history


def test_benchmark_respects_model_specific_precision_without_changing_other_models(monkeypatch):
    from ml_pipeline.pipelines_torch import benchmark
    flags = []

    def pipeline(**kwargs):
        flags.append(kwargs["use_mixed_precision"])
        return SimpleNamespace(model=kwargs["model"], fit=lambda X, y: [])

    monkeypatch.setattr(benchmark, "GeneralPipeline", pipeline)
    metadata = []
    monkeypatch.setattr(benchmark, "model_exists", lambda record, **kwargs: metadata.append(dict(record)) or False)
    monkeypatch.setattr(benchmark, "save_model", lambda *args, **kwargs: {"checkpoint_id": "fixture"})
    monkeypatch.setattr(benchmark, "save_metrics", lambda *args, **kwargs: None)
    configurations = [{"name": name, "class": torch.nn.Linear,
                       "params": {"in_features": 2, "out_features": 2}, **extra}
                      for name, extra in [("repair", {"use_mixed_precision": False}), ("unchanged", {})]]
    benchmark.BenchmarkRunner(model_configs=configurations, augmentations=[None], device="cpu",
                              use_kfold=False, epochs=1).run(np.ones((4, 2)), np.array([0, 1, 0, 1]))
    assert flags == [False, True]
    assert metadata[0]["use_mixed_precision"] is False
    assert "use_mixed_precision" not in metadata[1]
