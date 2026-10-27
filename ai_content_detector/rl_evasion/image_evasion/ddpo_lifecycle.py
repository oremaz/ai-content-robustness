"""Diffusion model setup, prompt partitions and resumable evaluation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .image_rewards import CompositeImageReward, ImageDetectorReward

logger = logging.getLogger("ai_content_detector.rl_evasion.image_evasion.ddpo_trainer")


def _diffusion_source(model_id):
    if not Path(model_id).is_dir() and os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}:
        from huggingface_hub import hf_hub_download
        return str(Path(hf_hub_download(model_id, "model_index.json", local_files_only=True)).parent)
    return model_id


class DDPOLifecycle:
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

    def _save_adapter(self, directory):
        save = getattr(self.pipeline.unet, "save_lora_adapter", self.pipeline.unet.save_pretrained)
        save(directory)

    def evaluate(self, num_samples: int = 50) -> Dict[str, float]:
        """Evaluate the trained model's evasion capability.

        Diffusion Purification (Saberi et al., ICLR 2024) is evaluated with a
        SEPARATE, pretrained `StableDiffusionImg2ImgPipeline` loaded from
        ``cfg.diffusion_model``. Reusing the trained attacker pipeline as the
        purifier would be circular ; the attacker could "unlearn" its own
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
        except Exception as e:  # noqa: BLE001 ; any loading failure is a skip signal
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
