"""DDPO-based image evasion trainer with full multi-step trajectory tracking.

Implements the full DDPO algorithm (Black et al., 2023) for RL fine-tuning of
diffusion models to evade AI-image detectors. Each denoising step is treated
as an action in a multi-step MDP, with per-step log-probabilities enabling
proper policy gradient updates via a PPO-style clipped objective.

MDP formulation:
    State:  s_t = (c, t, x_t) — context, timestep, noisy latent
    Action: a_t = x_{t-1}     — denoised latent
    Reward: r(x_0, c)         — sparse, only at final step (detector evasion score)

Per-step log-prob (Gaussian denoising kernel):
    log p_theta(x_{t-1} | x_t, c) = -0.5 * ||x_{t-1} - mu_theta||^2 / sigma_t^2

DDPO-IS objective (PPO-style clipping):
    L = -E[ sum_t min(ratio_t * A, clip(ratio_t, 1-eps, 1+eps) * A) ]
    ratio_t = exp(log_p_theta - log_p_theta_old)

References:
    - Black et al., "Training Diffusion Models with Reinforcement Learning" (2023)
    - Fan et al., DPOK (NeurIPS 2023)
    - kvablack/ddpo-pytorch reference implementation
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import ImageEvasionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-step log-probability computation
# ---------------------------------------------------------------------------


def ddim_step_with_logprob(
    scheduler,
    unet,
    latents: torch.Tensor,
    timestep: int,
    prompt_embeds: torch.Tensor,
    guidance_scale: float = 7.5,
    prev_sample: Optional[torch.Tensor] = None,
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
    t_input = torch.tensor([timestep] * 2, device=latents.device)

    noise_pred = unet(latent_input, t_input, encoder_hidden_states=prompt_embeds).sample

    # Classifier-free guidance split
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # DDIM parameters from scheduler
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[scheduler.previous_timestep(timestep)]
        if hasattr(scheduler, "previous_timestep")
        else scheduler.alphas_cumprod[max(timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 0)]
    )

    beta_prod_t = 1.0 - alpha_prod_t
    beta_prod_t_prev = 1.0 - alpha_prod_t_prev

    # Predicted x_0 from noise prediction
    pred_original = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()

    # Clip predicted x_0 for stability
    if scheduler.config.clip_sample:
        pred_original = pred_original.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)

    # DDIM predicted mean (mu_theta)
    mu = alpha_prod_t_prev.sqrt() * pred_original + beta_prod_t_prev.sqrt() * noise_pred

    # Variance (sigma_t^2) — DDIM uses eta * sigma for stochasticity
    # With eta=0 (deterministic DDIM), sigma=0 and log_prob is delta.
    # We use eta=1 (DDPM-like) for stochastic sampling needed by DDPO.
    sigma_sq = beta_prod_t_prev
    # Ensure minimum variance for numerical stability
    sigma_sq = torch.clamp(sigma_sq, min=1e-6)
    std = sigma_sq.sqrt()

    if prev_sample is not None:
        # Training phase: compute log_prob of the given sample under current policy
        sample = prev_sample
    else:
        # Sampling phase: draw from Gaussian
        noise = torch.randn_like(latents)
        sample = mu + std * noise

    # Gaussian log-probability: -0.5 * ||x - mu||^2 / sigma^2 - 0.5 * d * log(2pi * sigma^2)
    log_prob = -0.5 * ((sample - mu) ** 2 / sigma_sq).sum(dim=(1, 2, 3))
    log_prob -= 0.5 * sample[0].numel() * torch.log(2.0 * torch.pi * sigma_sq)

    return sample, log_prob


def pipeline_sample_with_logprob(
    pipeline,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
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
        dtype=prompt_embeds.dtype,
    )
    latents = latents * scheduler.init_noise_sigma

    scheduler.set_timesteps(num_inference_steps, device=device)

    all_latents = [latents.detach().clone()]
    all_log_probs = []

    for t in scheduler.timesteps:
        next_latents, log_prob = ddim_step_with_logprob(
            scheduler=scheduler,
            unet=unet,
            latents=latents,
            timestep=int(t),
            prompt_embeds=combined_embeds,
            guidance_scale=guidance_scale,
        )
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


# ---------------------------------------------------------------------------
# Per-prompt statistics tracker (for advantage normalization)
# ---------------------------------------------------------------------------


class PerPromptStatTracker:
    """Track running mean/std of rewards per prompt for advantage normalization.

    advantage = (reward - mean_reward_for_prompt) / (std_reward_for_prompt + eps)
    """

    def __init__(self, buffer_size: int = 32, min_count: int = 2):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self._stats: Dict[str, List[float]] = defaultdict(list)

    def update(self, prompts: List[str], rewards: List[float]) -> np.ndarray:
        """Record rewards and return normalized advantages."""
        advantages = np.zeros(len(rewards))

        for i, (prompt, reward) in enumerate(zip(prompts, rewards)):
            self._stats[prompt].append(reward)
            # Keep buffer bounded
            if len(self._stats[prompt]) > self.buffer_size:
                self._stats[prompt] = self._stats[prompt][-self.buffer_size:]

            buf = self._stats[prompt]
            if len(buf) >= self.min_count:
                advantages[i] = (reward - np.mean(buf)) / (np.std(buf) + 1e-8)
            else:
                advantages[i] = 0.0

        return advantages


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


class ImageDetectorReward:
    """Compute evasion reward from an image detector.

    reward = 1 - detector_score (higher = less detectable)
    """

    def __init__(self, detector, name: str = "detector"):
        self.detector = detector
        self.name = name

    def __call__(self, image) -> float:
        try:
            result = self.detector.detect(image)
            return 1.0 - result.score
        except Exception as e:
            logger.warning("Detector %s failed: %s", self.name, e)
            return 0.5


class CompositeImageReward:
    """Combine evasion, CLIP alignment, and aesthetic rewards."""

    def __init__(
        self,
        detector_rewards: List[ImageDetectorReward],
        weights: Optional[Dict[str, float]] = None,
    ):
        self.detector_rewards = detector_rewards
        self.weights = weights or {"evasion": 0.5, "clip": 0.3, "aesthetic": 0.2}
        self._clip_model = None
        self._clip_processor = None

    def _get_clip_score(self, image, prompt: str) -> float:
        """Compute CLIP text-image alignment score."""
        try:
            if self._clip_model is None:
                from transformers import CLIPProcessor, CLIPModel
                self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_model.eval()
                if torch.cuda.is_available():
                    self._clip_model = self._clip_model.cuda()

            inputs = self._clip_processor(text=[prompt], images=image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
            score = outputs.logits_per_image.item() / 100.0
            return min(max(score, 0.0), 1.0)
        except Exception:
            return 0.5

    def __call__(self, image, prompt: str = "") -> Dict[str, float]:
        evasion_scores = [dr(image) for dr in self.detector_rewards]
        mean_evasion = float(np.mean(evasion_scores)) if evasion_scores else 0.5

        clip_score = self._get_clip_score(image, prompt) if prompt else 0.5
        aesthetic_score = clip_score  # proxy; real impl would use LAION aesthetic predictor

        total = (
            self.weights["evasion"] * mean_evasion
            + self.weights["clip"] * clip_score
            + self.weights["aesthetic"] * aesthetic_score
        )

        return {
            "total": total,
            "evasion": mean_evasion,
            "clip": clip_score,
            "aesthetic": aesthetic_score,
            "per_detector": {dr.name: s for dr, s in zip(self.detector_rewards, evasion_scores)},
        }


# ---------------------------------------------------------------------------
# DDPO Trainer
# ---------------------------------------------------------------------------


class DDPOImageEvasionTrainer:
    """Train a diffusion model via full DDPO to produce images that evade detectors.

    Two-phase training loop per epoch:
    1. **Sample phase**: Generate images using current policy, store full
       denoising trajectories (latents, log_probs, rewards).
    2. **Train phase**: For each inner PPO epoch, iterate over timesteps,
       recompute current log_prob, compute importance ratio and clipped
       PPO loss, backprop through UNet.

    This is the full DDPO-IS algorithm (Black et al., 2023), not the
    simplified version that was here before.
    """

    def __init__(self, config: Optional[ImageEvasionConfig] = None):
        self.config = config or ImageEvasionConfig()
        self.pipeline = None
        self.reward_fn = None
        self.prompts = None
        self.stat_tracker = PerPromptStatTracker(
            buffer_size=getattr(self.config, "stat_tracking_buffer_size", 32),
        )

    def setup(self):
        """Load diffusion model, apply LoRA, set up rewards."""
        logger.info("Setting up DDPO image evasion trainer...")
        self._setup_model()
        self._setup_rewards()
        self._setup_prompts()
        logger.info("Setup complete.")

    def _setup_model(self):
        """Load Stable Diffusion with LoRA on UNet."""
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        from peft import LoraConfig

        cfg = self.config

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            cfg.diffusion_model,
            torch_dtype=torch.float16,
        )
        # Use DDIM scheduler for tractable log-prob computation
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config,
        )

        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to("cuda")

        # Apply LoRA to UNet
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            task_type=None,
        )
        self.pipeline.unet.add_adapter(lora_config)

        trainable = sum(p.numel() for p in self.pipeline.unet.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.pipeline.unet.parameters())
        logger.info("UNet LoRA: %d trainable / %d total (%.2f%%)", trainable, total, 100 * trainable / total)

    def _setup_rewards(self):
        """Build image detector reward ensemble."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from detectors.image_detectors import (
            EfficientNetDetector,
            CLIPImageDetector,
            SigLIPDetector,
        )

        detector_rewards = []
        detector_map = {
            "efficientnet_b4": EfficientNetDetector,
            "clip_classifier": CLIPImageDetector,
            "siglip_detector": SigLIPDetector,
        }

        for name in self.config.detector_names:
            if name in detector_map:
                try:
                    det = detector_map[name]()
                    if det.is_available():
                        detector_rewards.append(ImageDetectorReward(det, name=name))
                        logger.info("Loaded image detector reward: %s", name)
                except Exception as e:
                    logger.warning("Could not load detector %s: %s", name, e)

        self.reward_fn = CompositeImageReward(
            detector_rewards=detector_rewards,
            weights={
                "evasion": self.config.reward_evasion_weight,
                "clip": self.config.reward_clip_weight,
                "aesthetic": self.config.reward_aesthetic_weight,
            },
        )

    def _setup_prompts(self):
        """Load prompts for image generation."""
        cfg = self.config
        try:
            from datasets import load_dataset
            ds = load_dataset(cfg.prompt_dataset, split="train")
            col = next((c for c in ["Prompt", "prompt", "text"] if c in ds.column_names), ds.column_names[0])
            self.prompts = ds[col][:cfg.num_prompts]
        except Exception:
            self.prompts = [
                "A photograph of a cat sitting on a windowsill",
                "A landscape painting of mountains at sunset",
                "A portrait photograph of a woman smiling",
                "A still life photo of fruits on a table",
                "An aerial photograph of a city at night",
            ] * (cfg.num_prompts // 5)

        logger.info("Loaded %d prompts for training.", len(self.prompts))

    def _encode_prompts(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts into text embeddings for the pipeline."""
        device = self.pipeline.unet.device
        dtype = self.pipeline.unet.dtype

        text_inputs = self.pipeline.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            prompt_embeds = self.pipeline.text_encoder(text_inputs.input_ids)[0].to(dtype)

        # Unconditional embeddings for classifier-free guidance
        uncond_inputs = self.pipeline.tokenizer(
            [""] * len(prompts),
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            negative_embeds = self.pipeline.text_encoder(uncond_inputs.input_ids)[0].to(dtype)

        return prompt_embeds, negative_embeds

    def train(self):
        """Run full DDPO-IS training loop.

        Two-phase loop per epoch:
        1. Sample phase: generate images with current policy, store trajectories.
        2. Train phase: PPO updates over stored trajectories with clipped objective.
        """
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        clip_range = getattr(cfg, "clip_range", 1e-4)
        num_inner_epochs = getattr(cfg, "num_inner_epochs", 1)
        kl_coeff = cfg.kl_coeff

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.pipeline.unet.parameters()),
            lr=cfg.learning_rate,
        )

        device = self.pipeline.unet.device

        for epoch in range(cfg.num_train_epochs):
            logger.info("=== DDPO Epoch %d/%d ===", epoch + 1, cfg.num_train_epochs)

            # ---- Sample Phase ----
            self.pipeline.unet.eval()

            # Sample a batch of prompts
            batch_indices = np.random.permutation(len(self.prompts))[:cfg.sample_batch_size]
            batch_prompts = [self.prompts[i] for i in batch_indices]

            # Encode prompts
            prompt_embeds, negative_embeds = self._encode_prompts(batch_prompts)

            # Generate with trajectory tracking
            with torch.no_grad():
                final_latents, all_latents, all_log_probs = pipeline_sample_with_logprob(
                    pipeline=self.pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=7.5,
                )

            # Decode to images and compute rewards
            images = decode_latents_to_pil(self.pipeline, final_latents)
            rewards = []
            for img, prompt in zip(images, batch_prompts):
                r = self.reward_fn(img, prompt)
                rewards.append(r["total"])

            rewards_arr = np.array(rewards)

            # Compute per-prompt advantages
            if getattr(cfg, "per_prompt_stat_tracking", True):
                advantages = self.stat_tracker.update(batch_prompts, rewards)
            else:
                advantages = (rewards_arr - rewards_arr.mean()) / (rewards_arr.std() + 1e-8)

            advantages_t = torch.tensor(advantages, device=device, dtype=torch.float32)
            num_timesteps = len(all_log_probs)

            logger.info(
                "  Sample phase: mean_reward=%.4f, std_reward=%.4f",
                rewards_arr.mean(), rewards_arr.std(),
            )

            # ---- Train Phase ----
            self.pipeline.unet.train()

            for inner_epoch in range(num_inner_epochs):
                # Shuffle timestep order for each inner epoch
                timestep_order = np.random.permutation(num_timesteps)

                total_loss = 0.0
                total_kl = 0.0
                num_updates = 0

                for step_idx in timestep_order:
                    t = self.pipeline.scheduler.timesteps[step_idx]
                    latents_at_step = all_latents[step_idx].to(device)
                    next_latents = all_latents[step_idx + 1].to(device)
                    old_log_probs = all_log_probs[step_idx].to(device)

                    # Combined embeds for CFG
                    combined_embeds = torch.cat([negative_embeds, prompt_embeds])

                    # Recompute log-prob under current policy
                    _, current_log_probs = ddim_step_with_logprob(
                        scheduler=self.pipeline.scheduler,
                        unet=self.pipeline.unet,
                        latents=latents_at_step,
                        timestep=int(t),
                        prompt_embeds=combined_embeds,
                        guidance_scale=7.5,
                        prev_sample=next_latents,
                    )

                    # Importance ratio in log space
                    log_ratio = current_log_probs - old_log_probs
                    ratio = torch.exp(log_ratio)

                    # Clipped PPO objective
                    unclipped = -advantages_t * ratio
                    clipped = -advantages_t * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                    ppo_loss = torch.mean(torch.maximum(unclipped, clipped))

                    # KL penalty (optional regularization against base model)
                    kl_penalty = torch.mean(log_ratio ** 2) * 0.5  # approximate KL
                    loss = ppo_loss + kl_coeff * kl_penalty

                    loss.backward()

                    total_loss += ppo_loss.item()
                    total_kl += kl_penalty.item()
                    num_updates += 1

                    # Gradient accumulation
                    if num_updates % cfg.train_batch_size == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.pipeline.unet.parameters(), max_norm=1.0,
                        )
                        optimizer.step()
                        optimizer.zero_grad()

                # Flush remaining gradients
                if num_updates % cfg.train_batch_size != 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if num_updates > 0:
                    logger.info(
                        "  Inner epoch %d: ppo_loss=%.6f, kl=%.6f",
                        inner_epoch + 1, total_loss / num_updates, total_kl / num_updates,
                    )

            # Log epoch summary
            logger.info(
                "Epoch %d complete. Mean reward: %.4f",
                epoch + 1, rewards_arr.mean(),
            )

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint-epoch{epoch+1}")
                self.pipeline.unet.save_pretrained(ckpt_dir)
                logger.info("Checkpoint saved to %s", ckpt_dir)

        # Save final model
        final_dir = os.path.join(cfg.output_dir, "final")
        self.pipeline.unet.save_pretrained(final_dir)
        logger.info("DDPO training complete. Final model saved to %s", final_dir)

    def evaluate(self, num_samples: int = 50) -> Dict[str, float]:
        """Evaluate the trained model's evasion capability.

        Diffusion Purification (Saberi et al., ICML 2025) is evaluated with a
        SEPARATE, pretrained `StableDiffusionImg2ImgPipeline` loaded from
        ``cfg.diffusion_model``. Reusing the trained attacker pipeline as the
        purifier would be circular — the attacker could "unlearn" its own
        adversarial signature.
        """
        self.pipeline.unet.eval()
        results: Dict[str, List[float]] = {"total": [], "evasion": [], "clip": [], "purified_evasion": []}

        eval_prompts = self.prompts[-num_samples:] if len(self.prompts) > num_samples else self.prompts

        # Build a SEPARATE, pretrained purifier pipeline (no LoRA, no RL updates).
        # This guarantees the purification eval is not leaking the attacker's weights.
        purifier_pipe = None
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            purifier_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.config.diffusion_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
            ).to(self.pipeline.unet.device)
            assert id(purifier_pipe.unet) != id(self.pipeline.unet), (
                "Purifier must be a pretrained pipeline, not the attacker's."
            )
            purifier_pipe.set_progress_bar_config(disable=True)
        except Exception as e:  # noqa: BLE001 — any loading failure is a skip signal
            logger.warning(
                "Could not load independent purifier pipeline (%s). "
                "Skipping Diffusion Purification eval.", e,
            )
            purifier_pipe = None

        # Schedule is documented: forward-noise level strength=0.15 (Saberi et al., ICML 2025)
        # with 50 reverse DDIM steps over the same schedule as the attacker.
        purification_strength = 0.15
        purification_steps = 50

        for prompt in eval_prompts:
            with torch.no_grad():
                image = self.pipeline(prompt, num_inference_steps=50).images[0]

                purified_image = None
                if purifier_pipe is not None:
                    purified_image = purifier_pipe(
                        prompt=prompt, image=image,
                        strength=purification_strength,
                        num_inference_steps=purification_steps,
                    ).images[0]

            r = self.reward_fn(image, prompt)

            for k in ["total", "evasion", "clip"]:
                results[k].append(r[k])

            if purified_image is not None:
                r_purified = self.reward_fn(purified_image, prompt)
                results["purified_evasion"].append(r_purified["evasion"])

        if not results["purified_evasion"]:
            del results["purified_evasion"]

        return {f"mean_{k}": float(np.mean(v)) for k, v in results.items()}
