"""DDIM transition densities, reference KL and trajectory sampling."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def previous_ddim_timestep(scheduler, timestep):
    if hasattr(scheduler, "previous_timestep"):
        return int(scheduler.previous_timestep(timestep))
    return timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps


def ddim_step_with_logprob(
    scheduler,
    unet,
    latents: torch.Tensor,
    timestep: int,
    prompt_embeds: torch.Tensor,
    guidance_scale: float = 7.5,
    prev_sample: Optional[torch.Tensor] = None,
    eta: float = 1.0,
    generator: Optional[torch.Generator] = None,
    return_stats: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform one DDIM denoising step and return (next_latents, log_prob).

    The denoising step is modeled as a Gaussian:
        p_theta(x_{t-1} | x_t, c) = N(mu_theta, sigma_t^2 I)

    where mu_theta is the predicted mean from the UNet noise prediction and
    sigma_t is derived from the scheduler's alpha/beta schedule.

    If prev_sample is provided, compute log_prob of that sample under the
    current policy (for the training phase). Otherwise, sample and return
    both the sample and its log_prob (for the sampling phase).
    """
    # Classifier-free guidance: concat unconditional + conditional
    latent_input = torch.cat([latents] * 2)
    t_input = torch.full((latent_input.shape[0],), timestep, device=latents.device, dtype=torch.long)
    latent_input = scheduler.scale_model_input(latent_input, timestep) if hasattr(scheduler, "scale_model_input") else latent_input
    latent_input = latent_input.to(getattr(unet, "dtype", latents.dtype))

    noise_pred = unet(latent_input, t_input, encoder_hidden_states=prompt_embeds).sample

    # Classifier-free guidance split
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = (noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)).float()

    prev_timestep = previous_ddim_timestep(scheduler, timestep)
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(latents.device, torch.float32)
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep].to(latents.device, torch.float32)
        if prev_timestep >= 0 else scheduler.final_alpha_cumprod.to(latents.device, torch.float32)
    )
    beta_prod_t = 1.0 - alpha_prod_t
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        pred_original = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
        pred_epsilon = noise_pred
    elif prediction_type == "sample":
        pred_original = noise_pred
        pred_epsilon = (latents - alpha_prod_t.sqrt() * pred_original) / beta_prod_t.sqrt()
    elif prediction_type == "v_prediction":
        pred_original = alpha_prod_t.sqrt() * latents - beta_prod_t.sqrt() * noise_pred
        pred_epsilon = alpha_prod_t.sqrt() * noise_pred + beta_prod_t.sqrt() * latents
    else:
        raise ValueError(f"Unsupported DDIM prediction type: {prediction_type}")

    # Clip predicted x_0 for stability
    if scheduler.config.clip_sample:
        pred_original = pred_original.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)

    variance = scheduler._get_variance(timestep, prev_timestep).to(latents.device, torch.float32)
    std = eta * variance.sqrt()
    direction = (1.0 - alpha_prod_t_prev - std.square()).clamp(min=0.0).sqrt() * pred_epsilon
    mu = alpha_prod_t_prev.sqrt() * pred_original + direction

    if prev_sample is not None:
        # Training phase: compute log_prob of the given sample under current policy
        sample = prev_sample
    else:
        # Sampling phase: draw from Gaussian
        noise = torch.randn(
            latents.shape, device=latents.device, dtype=latents.dtype, generator=generator,
        )
        sample = mu + std * noise

    # Gaussian log-probability: -0.5 * ||x - mu||^2 / sigma^2 - 0.5 * d * log(2pi * sigma^2)
    if float(std) == 0:
        log_prob = torch.zeros(latents.shape[0], device=latents.device, dtype=torch.float32)
    else:
        std_float = std.float()
        log_prob = -(sample.detach().float() - mu.float()).square() / (2 * std_float.square())
        log_prob = log_prob - std_float.log() - 0.5 * np.log(2.0 * np.pi)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_stats:
        return sample, log_prob, mu.float(), std.float()
    return sample, log_prob


def initial_model_kl(unet, scheduler, latents, timestep, prompt_embeds, mean, std, reduction="mean", guidance_scale=7.5):
    """Conditional Gaussian KL per latent coordinate, current policy to initial SD."""
    if float(std) <= 0:
        raise ValueError("KL is undefined for a changed deterministic DDIM transition")
    unet.disable_adapters()
    try:
        with torch.no_grad():
            _, _, reference_mean, reference_std = ddim_step_with_logprob(
                scheduler, unet, latents, timestep, prompt_embeds,
                prev_sample=latents, return_stats=True, guidance_scale=guidance_scale,
            )
    finally:
        unet.enable_adapters()
    if not torch.equal(std, reference_std):
        raise ValueError("Initial and attacker transition variances differ")
    values = ((mean - reference_mean).square() / (2 * std.square())).flatten(1).mean(1)
    if reduction == "none":
        return values
    if reduction != "mean":
        raise ValueError("KL reduction must be mean or none")
    return values.mean()


@torch.no_grad()
def trajectory_initial_kl(pipeline, latents, prompt_embeds, negative_embeds):
    combined = torch.cat([negative_embeds, prompt_embeds])
    values = []
    for index, timestep in enumerate(pipeline.scheduler.timesteps):
        t = int(timestep)
        previous = previous_ddim_timestep(pipeline.scheduler, t)
        if float(pipeline.scheduler._get_variance(t, previous)) <= 0:
            continue
        current = latents[index].to(pipeline.unet.device)
        _, _, mean, std = ddim_step_with_logprob(
            pipeline.scheduler, pipeline.unet, current, t, combined,
            prev_sample=latents[index + 1].to(pipeline.unet.device), return_stats=True,
        )
        values.append(initial_model_kl(pipeline.unet, pipeline.scheduler, current, t, combined,
                                       mean, std, reduction="none"))
    if not values:
        raise ValueError("Reference KL needs at least one stochastic transition")
    return torch.stack(values).mean(0)


def pipeline_sample_with_logprob(
    pipeline,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    reference_kl_collector: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Run full denoising pipeline, returning trajectory of latents and log-probs.

    Returns:
        (final_latents, all_latents, all_log_probs)
        - all_latents[j]: latent at step j, shape (B, C, H, W)
        - all_log_probs[j]: log-prob at step j, shape (B,)
    """
    scheduler = pipeline.scheduler
    unet = pipeline.unet
    device = unet.device

    # Concatenate prompt embeds for classifier-free guidance
    combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Initial noise
    latents = torch.randn(
        (prompt_embeds.shape[0], unet.config.in_channels, 64, 64),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    latents = latents * scheduler.init_noise_sigma

    scheduler.set_timesteps(num_inference_steps, device=device)

    all_latents = [latents.detach().clone()]
    all_log_probs = []

    for t in scheduler.timesteps:
        result = ddim_step_with_logprob(
            scheduler=scheduler,
            unet=unet,
            latents=latents,
            timestep=int(t),
            prompt_embeds=combined_embeds,
            guidance_scale=guidance_scale,
            generator=generator,
            return_stats=reference_kl_collector is not None,
        )
        next_latents, log_prob = result[:2]
        if reference_kl_collector is not None and float(result[3]) > 0:
            values = initial_model_kl(unet, scheduler, latents, int(t), combined_embeds,
                                      result[2], result[3], reduction="none", guidance_scale=guidance_scale)
            reference_kl_collector.append(values.detach())
        all_latents.append(next_latents.detach().clone())
        all_log_probs.append(log_prob.detach())
        latents = next_latents

    return latents, all_latents, all_log_probs


def decode_latents_to_pil(pipeline, latents: torch.Tensor) -> List[Image.Image]:
    """Decode latent tensors to PIL images using the VAE."""
    with torch.no_grad():
        latents_scaled = latents / pipeline.vae.config.scaling_factor
        images = pipeline.vae.decode(latents_scaled.to(pipeline.vae.dtype)).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
