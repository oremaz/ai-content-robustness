from __future__ import annotations

import numpy as np
import pytest
import torch

from ai_content_detector.detectors.style_detector import StyleEmbeddingDetector


class Tokens(dict):
    def to(self, device):
        return Tokens({key: value.to(device) for key, value in self.items()})


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        batch = 1 if isinstance(texts, str) else len(texts)
        return Tokens({
            "input_ids": torch.ones((batch, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch, 4), dtype=torch.long),
        })


class FakeLUAR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.last_shape = None

    def forward(self, input_ids, attention_mask):
        self.last_shape = tuple(input_ids.shape)
        batch = input_ids.shape[0]
        return torch.ones((batch, 512), device=input_ids.device)


def _detector():
    detector = StyleEmbeddingDetector(device="cpu")
    detector._device = "cpu"
    detector._tokenizer = FakeTokenizer()
    detector._model = FakeLUAR()
    detector._available = True
    detector._load = lambda: None
    return detector


def test_luar_receives_batch_episode_sequence_shape():
    detector = _detector()
    embedding = detector._embed_single("text")
    assert detector._model.last_shape == (1, 1, 4)
    assert embedding.shape == (512,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0)


def test_batch_queries_each_form_one_episode():
    detector = _detector()
    embeddings = detector._embed_batch(["one", "two"])
    assert detector._model.last_shape == (2, 1, 4)
    assert embeddings.shape == (2, 512)


def test_detection_requires_explicit_support_set():
    detector = _detector()
    with pytest.raises(RuntimeError, match="setup_support_set"):
        detector.detect("query")


def test_empty_support_set_is_rejected():
    detector = _detector()
    with pytest.raises(ValueError, match="nonempty"):
        detector.setup_support_set([], ["human"])
