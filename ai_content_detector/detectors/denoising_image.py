"""Diffusion trajectory features for image detection."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

import numpy as np

from .ensemble import BaseDetector, DetectionResult
from PIL import Image
from .image_detectors_utils import _check_hf_image_config

logger = logging.getLogger("ai_content_detector.detectors.image_detectors")


class DenoisingTrajectoryDetector(BaseDetector):
    """DTAD using DDIM inversion and intermediate CLIP image features."""

    __module__ = "ai_content_detector.detectors.image_detectors"

    name = "Denoising Trajectory"
    modality = "image"

    def __init__(
        self,
        sd_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        clip_model_id: str = "openai/clip-vit-large-patch14",
        num_inference_steps: int = 50,
        trajectory_indices: Optional[Sequence[int]] = None,
        clip_layer: int = 15,
        device: str = "auto",
    ):
        self._sd_model_id = sd_model_id
        self._clip_model_id = clip_model_id
        self._num_inference_steps = int(num_inference_steps)
        self._trajectory_indices = None if trajectory_indices is None else tuple(trajectory_indices)
        self._clip_layer = int(clip_layer)
        self._device_spec = device
        self._pipeline = None
        self._clip_model = None
        self._clip_processor = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._pipeline is not None and self._clip_model is not None:
            return
        try:
            import torch
            from diffusers import DDIMScheduler, StableDiffusionPipeline
            from transformers import AutoProcessor, CLIPVisionModel

            device = self._device_spec
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            dtype = torch.float16 if device == "cuda" else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                self._sd_model_id, torch_dtype=dtype, safety_checker=None,
            ).to(device)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.set_progress_bar_config(disable=True)
            pipe.unet.eval()
            pipe.vae.eval()
            self._pipeline = pipe
            self._clip_processor = AutoProcessor.from_pretrained(self._clip_model_id)
            self._clip_model = CLIPVisionModel.from_pretrained(
                self._clip_model_id, torch_dtype=dtype,
            ).to(device).eval()
            with torch.no_grad():
                tokens = pipe.tokenizer(
                    "", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                    return_tensors="pt",
                ).to(device)
                self._uncond_embed = pipe.text_encoder(**tokens).last_hidden_state
            self._available = True
        except Exception as error:
            logger.warning("DenoisingTrajectoryDetector unavailable: %s", error)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from diffusers import StableDiffusionPipeline
                from transformers import CLIPVisionModel  # noqa: F401
                StableDiffusionPipeline.load_config(self._sd_model_id)
                self._available = _check_hf_image_config(self, self._clip_model_id)
            except Exception as error:
                self._availability_error = str(error)
                self._available = False
        return bool(self._available)

    def _to_latent(self, image: Image.Image) -> "torch.Tensor":
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((512, 512)), transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            distribution = self._pipeline.vae.encode(
                tensor.to(dtype=self._pipeline.vae.dtype),
            ).latent_dist
        return distribution.mean * self._pipeline.vae.config.scaling_factor

    @staticmethod
    def _ddim_inversion_update(latent, predicted_noise, alpha_current, alpha_next):
        predicted_clean = (
            latent - (1.0 - alpha_current).sqrt() * predicted_noise
        ) / alpha_current.sqrt().clamp(min=1e-8)
        next_latent = (
            alpha_next.sqrt() * predicted_clean
            + (1.0 - alpha_next).sqrt() * predicted_noise
        )
        return next_latent, predicted_clean

    def _decode_latents(self, latents: List["torch.Tensor"]) -> List[Image.Image]:
        import torch

        images = []
        scaling = self._pipeline.vae.config.scaling_factor
        for latent in latents:
            with torch.no_grad():
                decoded = self._pipeline.vae.decode(latent / scaling).sample
            array = decoded[0].float().cpu().permute(1, 2, 0).numpy()
            images.append(Image.fromarray(
                np.clip((array / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8),
            ))
        return images

    def _clip_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        import torch

        inputs = self._clip_processor(images=images, return_tensors="pt")
        pixels = inputs["pixel_values"].to(
            self._device, dtype=next(self._clip_model.parameters()).dtype,
        )
        with torch.no_grad():
            outputs = self._clip_model(pixel_values=pixels, output_hidden_states=True)
        if self._clip_layer >= len(outputs.hidden_states):
            raise ValueError(
                f"CLIP layer {self._clip_layer} unavailable; model has {len(outputs.hidden_states)} states"
            )
        embeddings = outputs.hidden_states[self._clip_layer][:, 0].float()
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.cpu().numpy()

    def detect(self, content: Union[Image.Image, np.ndarray]) -> DetectionResult:
        import torch

        self._load()
        if not self._available:
            raise RuntimeError("Denoising trajectory pipeline not available")
        if isinstance(content, np.ndarray):
            array = np.asarray(content)
            if array.ndim == 4:
                array = array[0]
            if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
                array = np.transpose(array, (1, 2, 0))
            if array.dtype != np.uint8:
                array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255).astype(np.uint8)
            image = Image.fromarray(array).convert("RGB")
        else:
            image = content.convert("RGB")

        latent = self._to_latent(image)
        scheduler = self._pipeline.scheduler
        scheduler.set_timesteps(self._num_inference_steps, device=self._device)
        timesteps = list(reversed(scheduler.timesteps))
        selected = set(range(len(timesteps)) if self._trajectory_indices is None else self._trajectory_indices)
        if any(index < 0 or index >= len(timesteps) for index in selected):
            raise ValueError("trajectory_indices contains an out-of-range inversion step")

        alphas = scheduler.alphas_cumprod.to(self._device, dtype=latent.dtype)
        current = latent
        denoising_outputs = []
        selected_timesteps = []
        for index, timestep in enumerate(timesteps):
            with torch.no_grad():
                predicted_noise = self._pipeline.unet(
                    current, timestep,
                    encoder_hidden_states=self._uncond_embed.to(latent.dtype),
                ).sample
            alpha_current = alphas[timestep]
            alpha_next = alphas[timesteps[index + 1]] if index + 1 < len(timesteps) else alpha_current
            current, predicted_clean = self._ddim_inversion_update(
                current, predicted_noise, alpha_current, alpha_next,
            )
            if index in selected:
                denoising_outputs.append(predicted_clean.detach())
                selected_timesteps.append(int(timestep.item()))

        trajectory_images = self._decode_latents(denoising_outputs)
        original = image.resize((512, 512), Image.BILINEAR)
        embeddings = self._clip_embeddings([original] + trajectory_images)
        similarities = embeddings[1:] @ embeddings[0]
        mean_similarity = float(similarities.mean())
        ai_score = float(np.clip((mean_similarity + 1.0) / 2.0, 0.0, 1.0))
        return DetectionResult(
            score=ai_score,
            label=DetectionResult.label_from_score(ai_score),
            detector_name=self.name,
            details={
                "mean_clip_similarity": mean_similarity,
                "summed_clip_similarity": float(similarities.sum()),
                "per_step_clip_similarity": similarities.tolist(),
                "timesteps": selected_timesteps,
                "sd_model_id": self._sd_model_id,
                "clip_model_id": self._clip_model_id,
                "clip_layer": self._clip_layer,
                "score_semantics": "uncalibrated_similarity_transform",
            },
        )

