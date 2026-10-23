from types import SimpleNamespace
import zlib

import numpy as np
import pytest
import torch
from scipy.stats import entropy, skew, kurtosis

from ai_content_detector.detectors.text_detectors import DivEyeDetector


def test_features_match_released_space_order_and_statistics():
    text = "A concrete example tests the exact feature order expected by the released classifier."
    s = np.array([1.0, 2.0, 4.0, 3.0, 7.0, 5.0], dtype=np.float32)
    second = np.diff(np.diff(-s))
    expected = [np.mean(s), np.std(s), np.var(s), skew(s), kurtosis(s),
                np.mean(np.diff(s)), np.std(np.diff(s)), np.var(second),
                entropy(np.histogram(second, bins=20, density=True)[0]),
                np.corrcoef(second[:-1], second[1:])[0, 1],
                len(zlib.compress(text.encode())) / len(text.encode())]
    np.testing.assert_array_equal(DivEyeDetector._div_features(s, text), expected)
    assert DivEyeDetector._div_features(s, text)[2] == np.var(s, ddof=0)


def test_biscope_uses_shifted_and_unshifted_loss_statistics_in_decile_order():
    forward = np.arange(1.0, 21.0)
    backward = forward[::-1]
    features = DivEyeDetector._biscope_features(forward, backward)
    assert features.shape == (72,)
    for p in range(1, 10):
        for offset, values in [(0, forward), (4, backward)]:
            suffix = values[len(values) * p // 10:]
            expected = [np.mean(suffix), np.max(suffix), np.min(suffix), np.std(suffix)]
            np.testing.assert_array_equal(features[(p - 1) * 8 + offset:(p - 1) * 8 + offset + 4], expected)


def test_constant_moments_preserve_release_nan_missing_value_semantics():
    features = DivEyeDetector._div_features(np.ones(15), "nonempty text")
    assert np.isnan(features[[3, 4, 9]]).all()
    assert not np.isinf(features).any()


def test_short_input_is_rejected_before_loading_models():
    detector = DivEyeDetector(device="cpu")
    detector._load = lambda: pytest.fail("Short input must not load models")
    with pytest.raises(ValueError, match="15 words"):
        detector.detect("short text")


def test_class_one_is_ai_and_details_identify_combined_release():
    detector = DivEyeDetector(device="cpu")
    detector.extract_features = lambda content: np.arange(83, dtype=float)
    detector._classifier = SimpleNamespace(predict_proba=lambda values: np.array([[0.1, 0.9]]))
    result = detector.detect("example")
    assert result.score == pytest.approx(0.9)
    assert result.details["feature_count"] == 83
    assert result.details["frozen"] is True
    assert result.details["variant"] == "authors_diveye_biscope_83_features"


def test_all_backbone_parameters_are_frozen_and_use_one_device(monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    loaded = []

    def model(source, **kwargs):
        item = torch.nn.Linear(2, 2)
        loaded.append((source, kwargs, item))
        return item

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    detector = DivEyeDetector(device="cpu")
    detector._load_classifier = lambda: None
    detector._load()
    assert len(loaded) == 2
    for source, kwargs, model in loaded:
        assert source in [value[0] for value in detector.BACKBONES.values()]
        assert kwargs["torch_dtype"] == torch.float32
        assert not model.training
        assert not any(parameter.requires_grad for parameter in model.parameters())
        assert next(model.parameters()).device.type == "cpu"
    detector.unload()
    assert detector._model is None and detector._bi_model is None


def test_wrong_classifier_schema_is_rejected(tmp_path, monkeypatch):
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=2, n_jobs=1)
    model.fit(np.arange(36).reshape(12, 3), [0, 1] * 6)
    path = tmp_path / "wrong.json"
    model.save_model(path)
    detector = DivEyeDetector(classifier_path=str(path))
    with pytest.raises(ValueError, match="83-feature"):
        detector._load_classifier()


def test_actual_83_feature_xgboost_checkpoint_loads_without_retraining(tmp_path):
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=2, n_jobs=1)
    features = np.random.default_rng(42).normal(size=(12, 83))
    model.fit(features, [0, 1] * 6)
    path = tmp_path / "matching.json"
    model.save_model(path)
    detector = DivEyeDetector(classifier_path=str(path))
    detector._load_classifier()
    np.testing.assert_array_equal(detector._classifier.predict_proba(features), model.predict_proba(features))


def test_feature_inference_uses_correct_token_alignment_and_no_gradients():
    detector = DivEyeDetector(device="cpu")
    detector._device = torch.device("cpu")
    detector._load = lambda: None
    forward_calls = []

    class Tokenizer:
        def encode(self, text, **kwargs):
            assert kwargs["max_length"] == 1024
            return torch.tensor([[1, 2, 3, 4, 1, 3]])

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1, 2]]) if text.startswith("Complete")
                                   else torch.tensor([[3, 1, 4, 2]]))

    class Model:
        def __call__(self, tokens=None, input_ids=None, **kwargs):
            assert not torch.is_grad_enabled()
            assert kwargs["use_cache"] is False
            ids = tokens if tokens is not None else input_ids
            logits = torch.arange(ids.shape[1] * 5, dtype=torch.float32).reshape(1, ids.shape[1], 5) / 7
            logits[:, :, 0] += torch.arange(ids.shape[1], dtype=torch.float32)
            forward_calls.append((ids, logits))
            return SimpleNamespace(logits=logits)

    detector._tokenizer = detector._bi_tokenizer = Tokenizer()
    detector._model = detector._bi_model = Model()
    features = detector.extract_features(" ".join(["example"] * 15))
    assert features.shape == (83,)
    ids, logits = forward_calls[1]
    target = ids[0, 2:]
    fce = torch.nn.functional.cross_entropy(logits[0, 1:-1], target, reduction="none").numpy()
    bce = torch.nn.functional.cross_entropy(logits[0, 2:], target, reduction="none").numpy()
    np.testing.assert_array_equal(features[11:], detector._biscope_features(fce, bce))
