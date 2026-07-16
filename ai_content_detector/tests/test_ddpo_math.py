from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from ai_content_detector.rl_evasion.image_evasion.ddpo_trainer import (
    CompositeImageReward,
    ImageDetectorReward,
    PerPromptStatTracker,
    ddim_step_with_logprob,
)


class Scheduler:
    def __init__(self):
        self.alphas_cumprod = torch.tensor([0.9, 0.8, 0.7, 0.6])
        self.final_alpha_cumprod = torch.tensor(1.0)
        self.config = types.SimpleNamespace(
            prediction_type="epsilon", clip_sample=False, clip_sample_range=1.0,
        )

    def previous_timestep(self, timestep):
        return timestep - 1

    def _get_variance(self, timestep, previous_timestep):
        alpha_t = self.alphas_cumprod[timestep]
        alpha_prev = self.alphas_cumprod[previous_timestep]
        return (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)


class UNet:
    def __call__(self, latent_input, timestep, encoder_hidden_states):
        return types.SimpleNamespace(sample=torch.full_like(latent_input, 0.4))


def test_ddim_mean_uses_scheduler_variance_without_double_counting():
    scheduler = Scheduler()
    latents = torch.ones((1, 1, 1, 1))
    alpha_t = scheduler.alphas_cumprod[3]
    alpha_prev = scheduler.alphas_cumprod[2]
    epsilon = torch.tensor(0.4)
    clean = (latents - (1 - alpha_t).sqrt() * epsilon) / alpha_t.sqrt()
    variance = scheduler._get_variance(3, 2)
    expected_mean = (
        alpha_prev.sqrt() * clean
        + (1 - alpha_prev - variance).sqrt() * epsilon
    )

    sample, log_prob = ddim_step_with_logprob(
        scheduler, UNet(), latents, 3, torch.zeros((2, 1, 1)),
        guidance_scale=1.0, prev_sample=expected_mean,
    )
    assert sample.item() == pytest.approx(expected_mean.item())
    assert torch.isfinite(log_prob).all()


def test_new_prompts_use_batch_advantages_instead_of_all_zero():
    tracker = PerPromptStatTracker(min_count=2)
    advantages = tracker.update(["a", "b"], [0.0, 1.0])
    assert np.allclose(advantages, [-1.0, 1.0], atol=1e-6)


def test_image_detector_failure_propagates():
    class Broken:
        def detect(self, image):
            raise RuntimeError("broken")

    with pytest.raises(RuntimeError, match="broken"):
        ImageDetectorReward(Broken())(object())


def test_composite_image_reward_rejects_empty_detector_list():
    with pytest.raises(ValueError, match="at least one detector"):
        CompositeImageReward([], weights={"evasion": 1.0, "clip": 0.0, "aesthetic": 0.0})
