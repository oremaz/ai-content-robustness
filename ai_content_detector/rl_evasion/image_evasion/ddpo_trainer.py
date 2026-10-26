"""DDPO-based image evasion trainer with full multi-step trajectory tracking.

Implements the full DDPO algorithm (Black et al., ICLR 2024) for RL fine-tuning of
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
    - Black et al., "Training Diffusion Models with Reinforcement Learning" (ICLR 2024)
    - Fan et al., DPOK (NeurIPS 2023)
    - kvablack/ddpo-pytorch reference implementation
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import ImageEvasionConfig

logger = logging.getLogger(__name__)


def _diffusion_source(model_id):
    if not Path(model_id).is_dir() and os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}:
        from huggingface_hub import hf_hub_download
        return str(Path(hf_hub_download(model_id, "model_index.json", local_files_only=True)).parent)
    return model_id


# ---------------------------------------------------------------------------
# Per-step log-probability computation
# ---------------------------------------------------------------------------


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


def reference_similarity_reward(kl, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Reference KL scale must be finite and positive")
    values = torch.as_tensor(kl)
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Reference KL must be finite and nonnegative")
    return torch.reciprocal(1 + values / scale)


def component_statistics(values, weight):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()),
            "std": float(values.std()), "weighted_std": float(weight * values.std()),
            "weighted_range": float(weight * np.ptp(values))}


def rollout_prompt_indices(count, size, repeats, seed):
    if repeats < 1 or size % repeats or size // repeats > count:
        raise ValueError("Rollout must contain whole prompt groups from the training pool")
    return np.repeat(np.random.default_rng(seed).permutation(count)[:size // repeats], repeats)


def grouped_advantages(prompts, rewards):
    prompts, rewards = np.asarray(prompts), np.asarray(rewards, dtype=float)
    advantages = np.zeros_like(rewards)
    for prompt in np.unique(prompts):
        mask = prompts == prompt
        if mask.sum() < 2:
            raise ValueError("Grouped advantages need at least two images per prompt")
        values = rewards[mask]
        advantages[mask] = (values - values.mean()) / (values.std() + 1e-8)
    return advantages


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

        rewards_arr = np.asarray(rewards, dtype=np.float64)
        global_std = float(rewards_arr.std())
        global_advantages = (
            (rewards_arr - rewards_arr.mean()) / (global_std + 1e-8)
            if global_std > 1e-8 else np.zeros_like(rewards_arr)
        )
        for i, (prompt, reward) in enumerate(zip(prompts, rewards)):
            self._stats[prompt].append(reward)
            # Keep buffer bounded
            if len(self._stats[prompt]) > self.buffer_size:
                self._stats[prompt] = self._stats[prompt][-self.buffer_size:]

            buf = self._stats[prompt]
            if len(buf) >= self.min_count:
                advantages[i] = (reward - np.mean(buf)) / (np.std(buf) + 1e-8)
            else:
                advantages[i] = global_advantages[i]

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
        result = self.detector.detect(image)
        if not 0.0 <= result.score <= 1.0:
            raise ValueError(f"Detector {self.name} returned invalid score {result.score}")
        return 1.0 - result.score


class CompositeImageReward:
    """Combine evasion, CLIP alignment, and aesthetic rewards."""

    def __init__(
        self,
        detector_rewards: List[ImageDetectorReward],
        weights: Optional[Dict[str, float]] = None,
        aesthetic_scorer=None,
        clip_model_id="openai/clip-vit-base-patch32",
    ):
        self.detector_rewards = detector_rewards
        self.weights = weights or {"evasion": 0.5, "clip": 0.3, "aesthetic": 0.2}
        if not detector_rewards:
            raise ValueError("CompositeImageReward requires at least one detector")
        if self.weights.get("aesthetic", 0.0) > 0 and aesthetic_scorer is None:
            raise ValueError("A nonzero aesthetic weight requires an aesthetic_scorer")
        self.aesthetic_scorer = aesthetic_scorer
        self.clip_model_id = clip_model_id
        self._clip_model = None
        self._clip_processor = None

    def _get_clip_score(self, image, prompt: str) -> float:
        """Compute CLIP text-image alignment score."""
        try:
            if self._clip_model is None:
                from transformers import CLIPProcessor, CLIPModel
                self._clip_model = CLIPModel.from_pretrained(self.clip_model_id)
                self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
                self._clip_model.eval()
                if torch.cuda.is_available():
                    self._clip_model = self._clip_model.cuda()

            inputs = self._clip_processor(text=[prompt], images=image, return_tensors="pt", truncation=True,
                                         max_length=self._clip_model.config.text_config.max_position_embeddings)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
            score = outputs.logits_per_image.item() / 100.0
            return min(max(score, 0.0), 1.0)
        except Exception as error:
            raise RuntimeError("CLIP alignment scoring failed") from error

    def __call__(self, image, prompt: str = "") -> Dict[str, float]:
        evasion_scores = [dr(image) for dr in self.detector_rewards]
        mean_evasion = float(np.mean(evasion_scores))

        clip_score = self._get_clip_score(image, prompt) if prompt else 0.5
        aesthetic_score = (
            float(self.aesthetic_scorer(image, prompt)) if self.aesthetic_scorer is not None else 0.0
        )

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

    This is the full DDPO-IS algorithm (Black et al., ICLR 2024), not the
    simplified version that was here before.
    """

    def __init__(self, config: Optional[ImageEvasionConfig] = None):
        self.config = config or ImageEvasionConfig()
        if self.config.kl_coeff and self.config.reward_reference_weight:
            raise ValueError("Choose a reference reward or a direct KL penalty, not both")
        if not 0 <= self.config.reward_reference_weight <= 1:
            raise ValueError("Reference reward weight must be in [0, 1]")
        if not np.isfinite(self.config.reference_kl_scale) or self.config.reference_kl_scale <= 0:
            raise ValueError("Reference KL scale must be finite and positive")
        self.pipeline = None
        self.reward_fn = None
        self.prompts = None
        self.stat_tracker = PerPromptStatTracker(
            buffer_size=getattr(self.config, "stat_tracking_buffer_size", 32),
            min_count=self.config.prompt_stat_min_count,
        )
        if self.config.advantage_mode not in {"legacy", "batch", "group"}:
            raise ValueError("Unknown image advantage mode")
        if self.config.rollout_batches < 1 or self.config.sample_batch_size < 1:
            raise ValueError("Rollout batch counts must be positive")

    def setup(self):
        """Load diffusion model, apply LoRA, set up rewards."""
        logger.info("Setting up DDPO image evasion trainer...")
        self._seed_everything()
        self._setup_model()
        self._setup_rewards()
        self._setup_prompts()
        logger.info("Setup complete.")

    def _seed_everything(self):
        from ..config import seed_everything
        seed_everything(self.config.seed)

    def _setup_model(self):
        """Load Stable Diffusion with LoRA on UNet."""
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        from peft import LoraConfig

        cfg = self.config

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            _diffusion_source(cfg.diffusion_model),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        # Use DDIM scheduler for tractable log-prob computation
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config,
        )

        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to("cuda")

        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        # Apply LoRA to UNet
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            task_type=None,
        )
        self.pipeline.unet.add_adapter(lora_config)
        for parameter in self.pipeline.unet.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.pipeline.unet.enable_gradient_checkpointing()

        trainable = sum(p.numel() for p in self.pipeline.unet.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.pipeline.unet.parameters())
        logger.info("UNet LoRA: %d trainable / %d total (%.2f%%)", trainable, total, 100 * trainable / total)

    def _setup_rewards(self):
        """Build image detector reward ensemble."""
        from ai_content_detector.detectors.image_detectors import (
            EfficientNetDetector,
            CLIPImageDetector,
        )
        from ai_content_detector.detectors.published_image import DDADetector

        detector_rewards = []
        detector_map = {
            "efficientnet_b4": EfficientNetDetector,
            "clip_classifier": CLIPImageDetector,
            "dda": lambda: DDADetector(checkpoint=self.config.dda_checkpoint, dda_source=self.config.dda_source),
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

        if not detector_rewards:
            raise RuntimeError(
                f"No requested image detector could be loaded: {self.config.detector_names}"
            )

        self.reward_fn = CompositeImageReward(
            detector_rewards=detector_rewards,
            clip_model_id=self.config.clip_reward_model,
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

        if len(self.prompts) < 5:
            raise ValueError("DDPO requires at least five prompts")
        rng = np.random.default_rng(cfg.seed)
        permutation = rng.permutation(len(self.prompts))
        n_eval = max(1, int(round(cfg.evaluation_fraction * len(permutation))))
        all_prompts = list(self.prompts)
        self.eval_prompts = [all_prompts[index] for index in permutation[:n_eval]]
        self.prompts = [all_prompts[index] for index in permutation[n_eval:]]

        logger.info(
            "Loaded %d training and %d immutable evaluation prompts.",
            len(self.prompts), len(self.eval_prompts),
        )

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
        from ..checkpointing import StopRequest
        with StopRequest() as self._stop_request:
            return self._train()

    def _train(self):
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
        generator = torch.Generator(device=device).manual_seed(cfg.seed)
        from dataclasses import asdict
        from ..checkpointing import (
            CheckpointClock, CheckpointPause, atomic_save, gradient_state,
            load_checkpoint, model_state, restore_gradients, restore_model,
            restore_rng, rng_state,
        )
        restart_path = os.path.join(cfg.output_dir, "restart.pt")
        restart = load_checkpoint(restart_path, cfg)
        self.training_history = [] if restart is None else restart.get("training_history", [])
        progress = {"epoch": 0, "inner_epoch": 0, "position": 0, "trajectory": None,
                    "timestep_order": None, "total_loss": 0.0, "total_kl": 0.0, "num_updates": 0}
        if restart is not None:
            restore_model(self.pipeline.unet, restart["model"])
            optimizer.load_state_dict(restart["optimizer"])
            restore_gradients(self.pipeline.unet, restart["gradients"])
            self.stat_tracker.__dict__.update(restart["stat_tracker"])
            self.prompts, self.eval_prompts = restart["prompts"], restart["eval_prompts"]
            progress = restart["progress"]
            restore_rng(restart["rng"], generator)
        clock = CheckpointClock()

        def save():
            atomic_save({
                "config": asdict(cfg), "model": model_state(self.pipeline.unet),
                "optimizer": optimizer.state_dict(), "gradients": gradient_state(self.pipeline.unet),
                "stat_tracker": self.stat_tracker.__dict__, "progress": progress,
                "training_history": self.training_history,
                "prompts": self.prompts, "eval_prompts": self.eval_prompts,
                "rng": rng_state(generator),
            }, restart_path)
            clock.saved()

        save()
        for epoch in range(progress["epoch"], cfg.num_train_epochs):
            logger.info("=== DDPO Epoch %d/%d ===", epoch + 1, cfg.num_train_epochs)

            if progress["trajectory"] is None:
                # ---- Sample Phase ----
                self.pipeline.unet.eval()

                # Sample a batch of prompts
                batch_indices = rollout_prompt_indices(len(self.prompts), cfg.sample_batch_size * cfg.rollout_batches,
                                                       cfg.samples_per_prompt, cfg.seed + epoch)
                batch_prompts = [self.prompts[i] for i in batch_indices]

                # Encode prompts
                chunks, components = [], []
                for start in range(0, len(batch_prompts), cfg.sample_batch_size):
                    prompts_chunk = batch_prompts[start:start + cfg.sample_batch_size]
                    prompt_embeds, negative_embeds = self._encode_prompts(prompts_chunk)
                    reference_kls = [] if cfg.reward_reference_weight or cfg.measure_reference_kl or cfg.measure_updated_reference_kl else None
                    with torch.no_grad():
                        final_latents, all_latents, all_log_probs = pipeline_sample_with_logprob(
                            pipeline=self.pipeline,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_embeds,
                            num_inference_steps=cfg.num_inference_steps,
                            guidance_scale=7.5,
                            generator=generator,
                            reference_kl_collector=reference_kls,
                        )
                    images = decode_latents_to_pil(self.pipeline, final_latents)
                    components.extend(self.reward_fn(img, prompt) for img, prompt in zip(images, prompts_chunk))
                    if reference_kls is not None and not reference_kls:
                        raise ValueError("No stochastic reference KL measurements were collected")
                    chunks.append({"prompt": prompt_embeds.detach().cpu(), "negative": negative_embeds.detach().cpu(),
                                   "latents": [v.detach().cpu() for v in all_latents],
                                   "log_probs": [v.detach().cpu() for v in all_log_probs],
                                   "kl": torch.stack(reference_kls).mean(0).cpu() if reference_kls is not None else None})
                prompt_embeds = torch.cat([c["prompt"] for c in chunks])
                negative_embeds = torch.cat([c["negative"] for c in chunks])
                all_latents = [torch.cat([c["latents"][t] for c in chunks]) for t in range(len(chunks[0]["latents"]))]
                all_log_probs = [torch.cat([c["log_probs"][t] for c in chunks]) for t in range(len(chunks[0]["log_probs"]))]
                rewards_arr = np.asarray([r["total"] for r in components], dtype=float)
                reward_statistics = {"base": component_statistics(rewards_arr, 1.0)}
                if chunks[0]["kl"] is not None:
                    raw_kl = torch.cat([c["kl"] for c in chunks])
                    similarity = reference_similarity_reward(raw_kl, cfg.reference_kl_scale).cpu().numpy()
                    rewards_arr = rewards_arr + cfg.reward_reference_weight * similarity
                    reward_statistics.update(reference=component_statistics(similarity, cfg.reward_reference_weight),
                                             raw_kl=component_statistics(raw_kl.cpu().numpy(), 1.0))
                for key, weight in (("evasion", cfg.reward_evasion_weight), ("clip", cfg.reward_clip_weight)):
                    if all(key in item for item in components):
                        reward_statistics[key] = component_statistics([item[key] for item in components], weight)
                if not np.isfinite(rewards_arr).all():
                    raise FloatingPointError("Nonfinite combined image reward")
                rewards = rewards_arr.tolist()

                # Compute per-prompt advantages
                if cfg.advantage_mode == "group":
                    advantages = grouped_advantages(batch_prompts, rewards)
                elif cfg.advantage_mode == "legacy" and getattr(cfg, "per_prompt_stat_tracking", True):
                    advantages = self.stat_tracker.update(batch_prompts, rewards)
                else:
                    advantages = (rewards_arr - rewards_arr.mean()) / (rewards_arr.std() + 1e-8)
                if cfg.advantage_clip is not None:
                    advantages = np.clip(advantages, -cfg.advantage_clip, cfg.advantage_clip)

                advantages_t = torch.tensor(advantages, device=device, dtype=torch.float32)
                num_timesteps = len(all_log_probs)

                logger.info(
                    "  Sample phase: mean_reward=%.4f, std_reward=%.4f",
                    rewards_arr.mean(), rewards_arr.std(),
                )

                progress["trajectory"] = {
                    "prompt_embeds": prompt_embeds.detach().cpu(),
                    "negative_embeds": negative_embeds.detach().cpu(),
                    "latents": [value.detach().cpu() for value in all_latents],
                    "log_probs": [value.detach().cpu() for value in all_log_probs],
                    "advantages": advantages_t.detach().cpu(), "rewards": rewards_arr,
                    "reward_statistics": reward_statistics,
                    "timesteps": self.pipeline.scheduler.timesteps.cpu(), "prompts": batch_prompts,
                }
                save()
            trajectory = progress["trajectory"]
            prompt_embeds = trajectory["prompt_embeds"].to(device)
            negative_embeds = trajectory["negative_embeds"].to(device)
            all_latents, all_log_probs = trajectory["latents"], trajectory["log_probs"]
            advantages_t, rewards_arr = trajectory["advantages"].to(device), trajectory["rewards"]
            self.pipeline.scheduler.set_timesteps(cfg.num_inference_steps, device=device)
            if not torch.equal(self.pipeline.scheduler.timesteps.cpu(), trajectory["timesteps"]):
                raise ValueError("DDPO scheduler timesteps differ from restart trajectory")
            num_timesteps = len(all_log_probs)
            if self._stop_request.requested:
                raise CheckpointPause("DDPO sampled trajectory and restart saved")

            # ---- Train Phase ----
            self.pipeline.unet.train()

            for inner_epoch in range(progress["inner_epoch"], num_inner_epochs):
                if progress["timestep_order"] is None:
                    stochastic = [index for index, timestep in enumerate(self.pipeline.scheduler.timesteps)
                                  if not hasattr(self.pipeline.scheduler, "_get_variance") or
                                  float(self.pipeline.scheduler._get_variance(
                                      int(timestep), previous_ddim_timestep(self.pipeline.scheduler, int(timestep)))) > 0]
                    steps = np.random.permutation(stochastic)
                    progress["timestep_order"] = [int(t) + batch * num_timesteps
                                                  for batch in range(cfg.rollout_batches) for t in steps]
                timestep_order = progress["timestep_order"]
                total_loss, total_kl, num_updates = (
                    progress["total_loss"], progress["total_kl"], progress["num_updates"]
                )

                for position in range(progress["position"], len(timestep_order)):
                    batch_idx, step_idx = divmod(int(timestep_order[position]), num_timesteps)
                    selection = slice(batch_idx * cfg.sample_batch_size, (batch_idx + 1) * cfg.sample_batch_size)
                    t = self.pipeline.scheduler.timesteps[step_idx]
                    latents_at_step = all_latents[step_idx][selection].to(device)
                    next_latents = all_latents[step_idx + 1][selection].to(device)
                    old_log_probs = all_log_probs[step_idx][selection].to(device)

                    # Combined embeds for CFG
                    combined_embeds = torch.cat([negative_embeds[selection], prompt_embeds[selection]])

                    # Recompute log-prob under current policy
                    step_kwargs = dict(
                        scheduler=self.pipeline.scheduler,
                        unet=self.pipeline.unet,
                        latents=latents_at_step,
                        timestep=int(t),
                        prompt_embeds=combined_embeds,
                        guidance_scale=7.5,
                        prev_sample=next_latents,
                    )
                    if kl_coeff:
                        _, current_log_probs, current_mean, transition_std = ddim_step_with_logprob(
                            **step_kwargs, return_stats=True,
                        )
                        kl_penalty = initial_model_kl(
                            self.pipeline.unet, self.pipeline.scheduler, latents_at_step, int(t),
                            combined_embeds, current_mean, transition_std,
                        )
                    else:
                        _, current_log_probs = ddim_step_with_logprob(**step_kwargs)
                        kl_penalty = current_log_probs.new_zeros(())

                    # Importance ratio in log space
                    log_ratio = current_log_probs - old_log_probs
                    ratio = torch.exp(log_ratio)

                    # Clipped PPO objective
                    unclipped = -advantages_t[selection] * ratio
                    clipped = -advantages_t[selection] * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                    ppo_loss = torch.mean(torch.maximum(unclipped, clipped))

                    loss = ppo_loss + kl_coeff * kl_penalty
                    if cfg.diagnostic_logging:
                        diagnostics = progress.setdefault("diagnostics", [])
                        diagnostics.append({"clip_fraction": float(((ratio - 1).abs() > clip_range).float().mean()),
                                            "log_ratio_abs": float(log_ratio.detach().abs().mean()),
                                            "approx_kl": float((torch.exp(log_ratio.detach()) - 1 - log_ratio.detach()).mean())})
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite DDPO objective; last safe restart retained")

                    accumulation = min(cfg.train_batch_size,
                                       len(timestep_order) - (position // cfg.train_batch_size) * cfg.train_batch_size)
                    (loss / accumulation).backward()

                    total_loss += ppo_loss.item()
                    total_kl += kl_penalty.item()
                    num_updates += 1

                    # Gradient accumulation
                    if num_updates % cfg.train_batch_size == 0:
                        norm = torch.nn.utils.clip_grad_norm_(
                            self.pipeline.unet.parameters(), max_norm=1.0, error_if_nonfinite=True,
                        )
                        if cfg.diagnostic_logging:
                            progress["diagnostics"][-1]["gradient_norm"] = float(norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    progress.update(position=position + 1, total_loss=total_loss,
                                    total_kl=total_kl, num_updates=num_updates)
                    if clock.due() or self._stop_request.requested:
                        save()
                        if self._stop_request.requested:
                            raise CheckpointPause("DDPO restart saved after denoising update")

                # Flush remaining gradients
                if num_updates % cfg.train_batch_size != 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.pipeline.unet.parameters(), max_norm=1.0, error_if_nonfinite=True,
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                if num_updates > 0:
                    record = {
                        "epoch": epoch + 1, "inner_epoch": inner_epoch + 1,
                        "mean_reward": float(rewards_arr.mean()),
                        "ppo_loss": total_loss / num_updates, "direct_kl_loss": total_kl / num_updates,
                        "initial_model_kl": trajectory.get("reward_statistics", {}).get("raw_kl", {}).get("mean", 0.0),
                        "reward_statistics": trajectory.get("reward_statistics", {}),
                        "reference_kl_scale": cfg.reference_kl_scale,
                        "sampled_images": len(rewards_arr),
                        "advantage_mean": float(advantages_t.mean()), "advantage_std": float(advantages_t.std(unbiased=False)),
                    }
                    if cfg.diagnostic_logging:
                        measurements = progress.pop("diagnostics", [])
                        record["optimization"] = {key: float(np.mean([v[key] for v in measurements if key in v]))
                                                   for key in {key for v in measurements for key in v}}
                    if cfg.measure_updated_reference_kl:
                        self.pipeline.unet.eval()
                        measured = trajectory_initial_kl(self.pipeline, all_latents, prompt_embeds, negative_embeds)
                        record["updated_initial_kl_per_sample"] = measured.cpu().tolist()
                        self.pipeline.unet.train()
                    self.training_history.append(record)
                    logger.info(
                        "  Inner epoch %d: ppo_loss=%.6f, reference_KL=%.6g, reference_reward=%.4f, direct_kl_loss=%.6g",
                        inner_epoch + 1, total_loss / num_updates, record["initial_model_kl"],
                        record["reward_statistics"].get("reference", {}).get("mean", 0.0), total_kl / num_updates,
                    )

                progress.update(inner_epoch=inner_epoch + 1, position=0, timestep_order=None,
                                total_loss=0.0, total_kl=0.0, num_updates=0)
                save()

            # Log epoch summary
            logger.info(
                "Epoch %d complete. Mean reward: %.4f",
                epoch + 1, rewards_arr.mean(),
            )

            # Save checkpoint
            callback = getattr(self, "validation_callback", None)
            if callback is not None and cfg.validation_interval and (epoch + 1) % cfg.validation_interval == 0:
                self.pipeline.unet.eval()
                with torch.no_grad():
                    callback(self, epoch + 1)
            if (epoch + 1) % cfg.adapter_save_interval == 0:
                ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint-epoch{epoch+1}")
                self._save_adapter(ckpt_dir)
                logger.info("Checkpoint saved to %s", ckpt_dir)

            progress.update(epoch=epoch + 1, inner_epoch=0, position=0, trajectory=None,
                            timestep_order=None, total_loss=0.0, total_kl=0.0, num_updates=0)
            save()
            if self._stop_request.requested:
                raise CheckpointPause("DDPO restart saved after epoch")

        # Save final model
        final_dir = os.path.join(cfg.output_dir, "final")
        self._save_adapter(final_dir)
        logger.info("DDPO training complete. Final model saved to %s", final_dir)

    def _save_adapter(self, directory):
        save = getattr(self.pipeline.unet, "save_lora_adapter", self.pipeline.unet.save_pretrained)
        save(directory)

    def evaluate(self, num_samples: int = 50) -> Dict[str, float]:
        """Evaluate the trained model's evasion capability.

        Diffusion Purification (Saberi et al., ICLR 2024) is evaluated with a
        SEPARATE, pretrained `StableDiffusionImg2ImgPipeline` loaded from
        ``cfg.diffusion_model``. Reusing the trained attacker pipeline as the
        purifier would be circular — the attacker could "unlearn" its own
        adversarial signature.
        """
        from ..checkpointing import StopRequest
        with StopRequest() as stop:
            return self._evaluate(num_samples, stop)

    def _evaluate(self, num_samples, stop):
        from dataclasses import asdict
        from pathlib import Path
        from ..checkpointing import (
            CheckpointPause, atomic_save, evaluation_state, rng_state,
        )
        self.pipeline.unet.eval()
        results: Dict[str, List[float]] = {"total": [], "evasion": [], "clip": [], "purified_evasion": []}

        eval_prompts = self.eval_prompts[:num_samples]

        # Build a SEPARATE, pretrained purifier pipeline (no LoRA, no RL updates).
        # This guarantees the purification eval is not leaking the attacker's weights.
        purifier_pipe = None
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            purifier_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                _diffusion_source(self.config.diffusion_model),
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

        # The independent purifier retains its pretrained img2img scheduler.
        purification_strength = 0.15
        purification_steps = 50

        path = Path(self.config.output_dir) / "evaluation_restart.pt"
        state = evaluation_state(path, {
            "config": asdict(self.config), "prompts": eval_prompts,
            "purifier_available": purifier_pipe is not None,
            "purification_strength": purification_strength,
            "purification_steps": purification_steps,
        }, self.pipeline.unet)
        for prompt in eval_prompts[len(state["scores"]):]:
            if stop.requested:
                raise CheckpointPause("DDPO evaluation saved")
            with torch.no_grad():
                image = self.pipeline(prompt, num_inference_steps=50).images[0]
                purified_image = None
                if purifier_pipe is not None:
                    purified_image = purifier_pipe(
                        prompt=prompt, image=image, strength=purification_strength,
                        num_inference_steps=purification_steps,
                    ).images[0]
            reward = self.reward_fn(image, prompt)
            purified_reward = self.reward_fn(purified_image, prompt) if purified_image is not None else None
            state["scores"].append({"reward": reward, "purified": purified_reward})
            state["rng"] = rng_state()
            atomic_save(state, path)
        if stop.requested:
            raise CheckpointPause("DDPO evaluation saved")
        for item in state["scores"]:
            for key in ("total", "evasion", "clip"):
                results[key].append(item["reward"][key])
            if item["purified"] is not None:
                results["purified_evasion"].append(item["purified"]["evasion"])
        if not results["purified_evasion"]:
            del results["purified_evasion"]
        return {f"mean_{key}": float(np.mean(values)) for key, values in results.items()}
