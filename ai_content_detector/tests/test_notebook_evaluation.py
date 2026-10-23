import json
from types import SimpleNamespace

import numpy as np
import pytest

from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
from scripts.notebook_evaluation import cached_probabilities


def test_inference_resumes_after_committed_batch_and_complete_cache_skips_model(tmp_path):
    inputs = ["first", "second", "third", "fourth", "fifth"]
    stop = SimpleNamespace(requested=False)
    calls = []

    class Model:
        def predict_proba(self, values):
            calls.append(list(values))
            stop.requested = True
            return np.tile([0.125, 0.875], (len(values), 1))

    with pytest.raises(CheckpointPause):
        cached_probabilities(Model, inputs, tmp_path, {"model_name": "fixture"}, stop, batch_size=2)
    progress_path = next(tmp_path.glob("*/progress.json"))
    assert json.loads(progress_path.read_text())["completed"] == 2
    stop.requested = False

    class ResumedModel:
        def predict_proba(self, values):
            calls.append(list(values))
            return np.tile([0.125, 0.875], (len(values), 1))

    probabilities = cached_probabilities(ResumedModel, inputs, tmp_path,
                                         {"model_name": "fixture"}, stop, batch_size=2)
    assert calls == [inputs[:2], inputs[2:4], inputs[4:]]
    assert np.array_equal(probabilities, np.tile([0.125, 0.875], (5, 1)))

    def must_not_load():
        raise AssertionError("Completed predictions must not load the model")

    assert np.array_equal(cached_probabilities(must_not_load, inputs, tmp_path,
                                               {"model_name": "fixture"}, stop, batch_size=2), probabilities)


def test_changed_inputs_or_checkpoint_do_not_reuse_probabilities(tmp_path):
    stop = SimpleNamespace(requested=False)
    calls = []

    class Model:
        def predict_proba(self, values):
            calls.append(list(values))
            return np.tile([0.3333333333333, 0.6666666666667], (len(values), 1))

    for inputs, checkpoint in [(["one"], "a"), (["different"], "a"), (["one"], "b")]:
        result = cached_probabilities(Model, inputs, tmp_path, {"checkpoint": checkpoint}, stop)
        assert result[0, 0] == 0.3333333333333
    assert len(calls) == 3
    assert len(list(tmp_path.glob("*/progress.json"))) == 3


def test_invalid_probabilities_are_never_committed(tmp_path):
    model = SimpleNamespace(predict_proba=lambda values: np.array([[np.nan, 0.5]]))
    with pytest.raises(ValueError, match="Invalid notebook probabilities"):
        cached_probabilities(lambda: model, ["sample"], tmp_path, {}, SimpleNamespace(requested=False))
    assert not list(tmp_path.glob("*/progress.json"))
