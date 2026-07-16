from __future__ import annotations

import types

import pytest
import torch

from ai_content_detector.rl_evasion.text_evasion.proxy_evasion import ProxyEvasionWrapper


class Model(torch.nn.Module):
    def __init__(self, logits, vocab_size=3, eos_token_id=2):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.values = torch.tensor(logits, dtype=torch.float32)
        self.config = types.SimpleNamespace(vocab_size=vocab_size, eos_token_id=eos_token_id)

    def forward(self, input_ids):
        batch, length = input_ids.shape
        logits = self.values.expand(batch, length, -1) + self.anchor
        return types.SimpleNamespace(logits=logits)


def test_identical_proxy_and_reference_leave_target_distribution_unchanged(monkeypatch):
    target = Model([10.0, -10.0, -10.0])
    proxy = Model([1.0, 2.0, 3.0])
    reference = Model([1.0, 2.0, 3.0])
    wrapper = ProxyEvasionWrapper(target, proxy, reference, intervention_alpha=10.0)
    monkeypatch.setattr(torch, "multinomial", lambda probabilities, num_samples: probabilities.argmax(dim=-1, keepdim=True))
    output = wrapper.generate(torch.tensor([[1]]), max_new_tokens=1)
    assert output[0, -1].item() == 0


def test_vocab_mismatch_is_rejected():
    with pytest.raises(ValueError, match="share a vocabulary"):
        ProxyEvasionWrapper(Model([1, 2, 3]), Model([1, 2], vocab_size=2), Model([1, 2, 3]))
