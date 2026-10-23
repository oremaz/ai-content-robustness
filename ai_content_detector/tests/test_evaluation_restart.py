import sys
from types import SimpleNamespace

import pytest
import torch
from transformers.tokenization_utils_base import BatchEncoding

from ai_content_detector.rl_evasion import checkpointing
from ai_content_detector.rl_evasion.config import ImageEvasionConfig, TextEvasionConfig, seed_everything


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    @property
    def device(self):
        return self.weight.device

    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, torch.randint(3, 100, (1, 3))], dim=1)


class Tokenizer:
    def __call__(self, *args, **kwargs):
        return BatchEncoding({"input_ids": torch.tensor([[1, 2]])})

    def decode(self, values, **kwargs):
        return str(values.tolist())


def text_trainer(path):
    from ai_content_detector.rl_evasion.text_evasion.grpo_trainer import GRPOTextEvasionTrainer

    trainer = GRPOTextEvasionTrainer(TextEvasionConfig(output_dir=str(path)))
    trainer.model, trainer.tokenizer = Policy(), Tokenizer()
    trainer.prompts = trainer.eval_prompts = [f"prompt {i}" for i in range(5)]
    trainer.references = trainer.eval_references = [f"source {i}" for i in range(5)]
    trainer.reward_fn = lambda source, output: {
        "total": float(torch.rand(())), "evasion": float(torch.rand(())),
        "semantic": 0.9, "per_detector": {"fixture": float(torch.rand(()))},
    }
    return trainer


@pytest.mark.parametrize("phase", ["generated", "scores"])
def test_text_evaluation_retains_samples_scores_and_rng(tmp_path, monkeypatch, phase):
    seed_everything(42)
    expected = text_trainer(tmp_path / "full").evaluate()
    seed_everything(42)
    original = checkpointing.atomic_save

    def interrupt(state, path):
        original(state, path)
        if len(state[phase]) == 2:
            raise checkpointing.CheckpointPause("interrupted evaluation")

    monkeypatch.setattr(checkpointing, "atomic_save", interrupt)
    with pytest.raises(checkpointing.CheckpointPause):
        text_trainer(tmp_path / "resume").evaluate()
    monkeypatch.setattr(checkpointing, "atomic_save", original)
    torch.rand(100)
    assert text_trainer(tmp_path / "resume").evaluate() == expected
    completed = text_trainer(tmp_path / "resume")
    completed.model.generate = lambda **kwargs: pytest.fail("generated a completed sample")
    completed.reward_fn = lambda *args: pytest.fail("rescored a completed sample")
    assert completed.evaluate() == expected
    with torch.no_grad():
        completed.model.weight.add_(1)
    with pytest.raises(ValueError, match="configuration differs"):
        completed.evaluate()


class ImagePipeline:
    def __init__(self):
        self.unet = Policy()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, device):
        return self

    def set_progress_bar_config(self, **kwargs):
        pass

    def __call__(self, prompt, **kwargs):
        return SimpleNamespace(images=[float(torch.rand(()))])


def image_trainer(path):
    from ai_content_detector.rl_evasion.image_evasion.ddpo_trainer import DDPOImageEvasionTrainer

    trainer = DDPOImageEvasionTrainer(ImageEvasionConfig(output_dir=str(path)))
    trainer.pipeline = ImagePipeline()
    trainer.eval_prompts = [f"prompt {i}" for i in range(5)]
    trainer.reward_fn = lambda image, prompt: {
        "total": float(torch.rand(())), "evasion": image, "clip": 0.8,
    }
    return trainer


def test_image_evaluation_retains_purification_scores_and_rng(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(StableDiffusionImg2ImgPipeline=ImagePipeline))
    seed_everything(42)
    expected = image_trainer(tmp_path / "full").evaluate()
    assert "mean_purified_evasion" in expected
    seed_everything(42)
    original = checkpointing.atomic_save

    def interrupt(state, path):
        original(state, path)
        if len(state["scores"]) == 2:
            raise checkpointing.CheckpointPause("interrupted evaluation")

    monkeypatch.setattr(checkpointing, "atomic_save", interrupt)
    with pytest.raises(checkpointing.CheckpointPause):
        image_trainer(tmp_path / "resume").evaluate()
    monkeypatch.setattr(checkpointing, "atomic_save", original)
    torch.rand(100)
    assert image_trainer(tmp_path / "resume").evaluate() == expected
    completed = image_trainer(tmp_path / "resume")
    completed.reward_fn = lambda *args: pytest.fail("rescored a completed image")
    assert completed.evaluate() == expected


def test_offline_diffusion_uses_pinned_component_snapshot(tmp_path, monkeypatch):
    from ai_content_detector.rl_evasion.image_evasion.ddpo_trainer import _diffusion_source
    import huggingface_hub

    index = tmp_path / "snapshots/revision/model_index.json"
    calls = []

    def cached_file(repo_id, filename, local_files_only):
        calls.append((repo_id, filename, local_files_only))
        return str(index)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", cached_file)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert _diffusion_source("author/diffusion-model") == str(index.parent)
    assert calls == [("author/diffusion-model", "model_index.json", True)]
    monkeypatch.delenv("HF_HUB_OFFLINE")
    assert _diffusion_source("author/diffusion-model") == "author/diffusion-model"
