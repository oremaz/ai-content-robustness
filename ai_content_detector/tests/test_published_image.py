import types
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from ai_content_detector.detectors.published_image import CommunityForensicsDetector, DDADetector
from ai_content_detector.rl_evasion.image_evasion.ddpo_trainer import ddim_step_with_logprob, initial_model_kl
from ai_content_detector.rl_evasion.image_evasion.research import metrics, partition_rows, split_defender_rows


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.tensor(0.2))
        self.enabled = True

    def disable_adapters(self):
        self.enabled = False

    def enable_adapters(self):
        self.enabled = True

    def forward(self, latents, timestep, encoder_hidden_states):
        assert len(timestep) == len(latents)
        noise = 0.4 + (self.delta if self.enabled else 0.0)
        return types.SimpleNamespace(sample=torch.ones_like(latents) * noise)


def test_initial_kl_uses_disabled_initial_adapter_and_has_gradient():
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler(clip_sample=False)
    scheduler.set_timesteps(50)
    policy = Policy()
    latents, embeddings = torch.ones(3, 1, 2, 2), torch.zeros(6, 1, 1)
    timestep = int(scheduler.timesteps[5])
    _, log_prob, mean, std = ddim_step_with_logprob(scheduler, policy, latents, timestep, embeddings,
                                                   prev_sample=latents, return_stats=True)
    kl = initial_model_kl(policy, scheduler, latents, timestep, embeddings, mean, std)
    assert kl > 0 and torch.isfinite(log_prob).all()
    kl.backward()
    assert policy.delta.grad > 0 and policy.enabled
    policy.delta.data.zero_()
    _, _, mean, std = ddim_step_with_logprob(scheduler, policy, latents, timestep, embeddings,
                                            prev_sample=latents, return_stats=True)
    assert initial_model_kl(policy, scheduler, latents, timestep, embeddings, mean, std).item() == 0


def test_terminal_ddim_transition_has_no_fake_gaussian_density():
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler(clip_sample=False)
    scheduler.set_timesteps(50)
    latents, embeddings = torch.ones(2, 1, 2, 2), torch.zeros(4, 1, 1)
    _, probability, mean, std = ddim_step_with_logprob(scheduler, Policy(), latents, int(scheduler.timesteps[-1]),
                                                       embeddings, return_stats=True)
    assert std == 0 and probability.tolist() == [0, 0]
    with pytest.raises(ValueError, match="deterministic"):
        initial_model_kl(Policy(), scheduler, latents, 0, embeddings, mean, std)


def test_race_resolution_matches_real_and_generated_images():
    detector = DDADetector(device="cpu", input_size=200)
    small = Image.new("RGB", (200, 200), "white")
    large = Image.new("RGB", (512, 512), "white")
    torch.testing.assert_close(detector.transform(small), detector.transform(large))
    assert detector.transform(small).shape == (3, 336, 336)


@pytest.mark.parametrize("size", [224, 384])
def test_community_native_preprocessing_and_fake_score(size):
    detector = CommunityForensicsDetector(f"OwensLab/commfor-model-{size}", device="cpu")
    detector._model = Mock(return_value=torch.tensor([[2.0], [-2.0]]))
    values = detector.predict_batch([Image.new("RGB", (300, 220))] * 2)
    torch.testing.assert_close(torch.tensor(values), torch.tensor([2.0, -2.0]).sigmoid())
    assert detector._model.call_args.args[0].shape == (2, 3, size, size)


def test_community_missing_checkpoint_is_unavailable_without_loading(tmp_path):
    detector = CommunityForensicsDetector(checkpoint=tmp_path / "missing.safetensors", device="cpu")
    assert not detector.is_available()
    assert detector._model is None


def test_artifact_partition_reproduces_existing_sampling_without_overlap():
    samples = [(Path(f"source/{i}.png"), i % 2) for i in range(1000)]
    fitting, evaluation = partition_rows(samples)
    assert len(fitting) == 120 and len(evaluation) == 30
    assert not {row['path'] for row in fitting} & {row['path'] for row in evaluation}
    assert (fitting, evaluation) == partition_rows(samples)
    assert all(row["label"] == 1 - int(Path(row["path"]).stem) % 2 for row in fitting + evaluation)


def test_defender_split_reserves_the_same_prompt_positions():
    fake = [{"path": f"fake/{i}", "sample_id": str(i)} for i in range(10)]
    real = [{"path": f"real/{i}"} for i in range(10)]
    train, validation = split_defender_rows(fake, real)
    assert [row['sample_id'] for row in train[:8]] == [str(i) for i in range(8)]
    assert len(validation) == 4
    with pytest.raises(ValueError, match="overlap"):
        split_defender_rows([{"path": "same"}] * 10, real)


def test_image_metrics_have_fake_positive_direction_and_human_fpr():
    rows = [{"label": 1}, {"label": 1}, {"label": 0}, {"label": 0}]
    result = metrics(rows, [0.9, 0.6, 0.8, 0.1])
    assert result['ai_detection_rate'] == 1.0
    assert result['human_false_positive_rate'] == 0.5
    assert result['roc_auc'] == 0.75
    with pytest.raises(ValueError, match="finite"):
        metrics(rows, [0.9, np.nan, 0.8, 0.1])
