"""DDPO optimization, restart progress and historical public imports.

Transition densities and trajectory sampling live in diffusion_policy.py;
model setup and resumable evaluation live in ddpo_lifecycle.py.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch

from ..config import ImageEvasionConfig

from .ddpo_lifecycle import DDPOLifecycle, _diffusion_source
from .diffusion_policy import (
    previous_ddim_timestep, ddim_step_with_logprob, initial_model_kl,
    trajectory_initial_kl, pipeline_sample_with_logprob, decode_latents_to_pil,
)
from .image_rewards import (
    reference_similarity_reward, component_statistics, rollout_prompt_indices,
    grouped_advantages, PerPromptStatTracker, ImageDetectorReward, CompositeImageReward,
)


logger = logging.getLogger(__name__)


class DDPOImageEvasionTrainer(DDPOLifecycle):
    """Train a diffusion model via full DDPO to produce images that evade detectors.

    Two-phase training loop per epoch:
    1. **Sample phase**: Generate images using current policy, store full
       denoising trajectories (latents, log_probs, rewards).
    2. **Train phase**: For each inner PPO epoch, iterate over timesteps,
       recompute current log_prob, compute importance ratio and clipped
       PPO loss, backprop through UNet.
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




for _export in (
    previous_ddim_timestep, ddim_step_with_logprob, initial_model_kl,
    trajectory_initial_kl, pipeline_sample_with_logprob, decode_latents_to_pil,
    reference_similarity_reward, component_statistics, rollout_prompt_indices,
    grouped_advantages, PerPromptStatTracker, ImageDetectorReward, CompositeImageReward,
    _diffusion_source,
):
    _export.__module__ = __name__
del _export
