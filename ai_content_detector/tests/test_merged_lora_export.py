"""Native exports must preserve adapter predictions and leave training intact."""

import importlib.metadata

import pytest
import torch

from ai_content_detector.watermarking.backbones import inject_lora, merged_lora_state_for_generator


@pytest.mark.parametrize("architecture", ["rar", "uvit"])
def test_merged_export_matches_live_adapter_without_mutation(monkeypatch, architecture):
    monkeypatch.setattr(importlib.metadata, "packages_distributions", lambda: {})

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.GELU(), torch.nn.Linear(7, 3))
            self.register_buffer("counter", torch.tensor(3))

        def get_trainable_modules(self):
            return torch.nn.ModuleDict({"uvit": self})

        def forward(self, x):
            return self.layers(x)

    torch.manual_seed(182)
    model = Network().requires_grad_(False)
    inject_lora(model, 2)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.03)
    model.eval()
    inputs = torch.randn(4, 5)
    expected = model(inputs)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    exported = merged_lora_state_for_generator(model, architecture)
    restored = Network().eval()
    target = restored.get_trainable_modules() if architecture == "uvit" else restored
    target.load_state_dict(exported, strict=True)
    torch.testing.assert_close(restored(inputs), expected, rtol=2.e-6, atol=2.e-7)
    assert not any("lora_" in key or "base_layer" in key for key in exported)
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    from ai_content_detector.rl_evasion.checkpointing import model_state, restore_model
    saved = model_state(model)
    assert set(saved) == {name for name, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(1.)
    restore_model(model, saved)
    torch.testing.assert_close(model(inputs), expected)
    torch.nn.functional.mse_loss(model(inputs), torch.zeros_like(expected)).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
