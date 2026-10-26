"""Released DDA and Community Forensics classifiers with native preprocessing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from .ensemble import BaseDetector, DetectionResult

RESOURCES = Path(__file__).resolve().parents[2] / "results/research_workflow/image_dda/resources"
COMMUNITY_REVISIONS = {"OwensLab/commfor-model-224": "26afc31e6b40c312c3fd42c05a758be62446215b",
                       "OwensLab/commfor-model-384": "6076002bf0d9dd37537f965ee2f06f826c333b61"}


def dda_resources():
    manifest = json.loads((RESOURCES / "provenance.json").read_text())
    return (manifest["checkpoint"]["files"]["DDA_ckpt.pth"]["path"],
            manifest["dinov2"]["path"], manifest["Dual-Data-Alignment"]["path"])


def dda_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.CenterCrop(336), transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])


class DDAClassifier(nn.Module):
    def __init__(self, dinov2_source, dda_source):
        super().__init__()
        lora_path = Path(dda_source) / "Inference/models/lora.py"
        spec = importlib.util.spec_from_file_location("_released_dda_lora", lora_path)
        lora = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lora)
        backbone = torch.hub.load(str(dinov2_source), "dinov2_vitl14", source="local", pretrained=False)
        backbone.requires_grad_(False)
        self.base_model = nn.Module()
        self.base_model.model = lora.apply_lora_to_linear_layers(
            backbone, rank=8, alpha=1.0,
            target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        )
        self.base_model.fc = nn.Linear(1024, 1)

    def forward(self, images):
        features = self.base_model.model.forward_features(images)["x_norm_clstoken"]
        return self.base_model.fc(features).flatten()


class DDADetector(BaseDetector):
    name = "DDA"
    modality = "image"

    def __init__(self, checkpoint=None, dinov2_source=None, dda_source=None, device="auto", input_size=None):
        self.checkpoint = checkpoint
        self.dinov2_source = dinov2_source
        self.dda_source = dda_source
        self.input_size = input_size
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self._model = None
        self._transform = dda_transform()

    def _paths(self):
        if self.checkpoint and self.dinov2_source and self.dda_source:
            return self.checkpoint, self.dinov2_source, self.dda_source
        defaults = dda_resources()
        return (self.checkpoint or defaults[0], self.dinov2_source or defaults[1], self.dda_source or defaults[2])

    def is_available(self):
        try:
            return all(Path(path).exists() for path in self._paths())
        except (OSError, KeyError, ValueError) as error:
            self._availability_error = str(error)
            return False

    def _load(self):
        if self._model is None:
            checkpoint, dinov2_source, dda_source = self._paths()
            model = DDAClassifier(dinov2_source, dda_source)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if "base_checkpoint" in state:
                base = torch.load(state["base_checkpoint"], map_location="cpu", weights_only=False)
                model.load_state_dict(base["model"], strict=True)
                expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
                if set(state["model"]) != expected:
                    raise ValueError("Adapted DDA checkpoint does not contain exactly the LoRA and head weights")
                model.load_state_dict(state["model"], strict=False)
            else:
                model.load_state_dict(state["model"], strict=True)
            self._model = model.to(self.device).eval()

    def transform(self, image):
        image = image.convert("RGB")
        if self.input_size is not None:
            image = image.resize((self.input_size, self.input_size), Image.Resampling.BICUBIC)
        return self._transform(image)

    def predict_logits(self, images):
        self._load()
        inputs = torch.stack([self.transform(image) for image in images]).to(self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            scores = self._model(inputs).float().cpu().numpy()
        if not np.isfinite(scores).all():
            raise FloatingPointError("DDA produced nonfinite logits")
        return scores

    def predict_batch(self, images):
        return torch.from_numpy(self.predict_logits(images)).sigmoid().numpy()

    def detect(self, content):
        if not isinstance(content, Image.Image):
            content = Image.fromarray(np.asarray(content))
        score = float(self.predict_batch([content])[0])
        return DetectionResult(score=score, label="ai" if score >= 0.5 else "human",
                               detector_name=self.name,
                               details={"score_semantics": "classifier_probability", "ai_class": 1,
                                        "checkpoint": str(self._paths()[0]), "crop_size": 336})


class CommunityForensicsDetector(BaseDetector):
    name = "Community Forensics"
    modality = "image"

    def __init__(self, model_id="OwensLab/commfor-model-384", revision=None, checkpoint=None, device="auto"):
        if model_id not in {"OwensLab/commfor-model-224", "OwensLab/commfor-model-384"}:
            raise ValueError("Use an official Community Forensics 224 or 384 checkpoint")
        self.model_id, self.revision, self.checkpoint = model_id, revision or COMMUNITY_REVISIONS[model_id], checkpoint
        self.size = int(model_id.rsplit("-", 1)[1])
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self._model = None
        from torchvision import transforms
        self._transform = transforms.Compose([
            transforms.Resize(256 if self.size == 224 else 440), transforms.CenterCrop(self.size),
            transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _weight_path(self, local_files_only=False):
        if self.checkpoint is not None:
            return self.checkpoint
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.model_id, "model.safetensors", revision=self.revision,
                               local_files_only=local_files_only)

    def is_available(self):
        try:
            return Path(self._weight_path(local_files_only=True)).is_file()
        except (OSError, ValueError) as error:
            self._availability_error = str(error)
            return False

    def _load(self):
        if self._model is None:
            import timm
            from safetensors.torch import load_file
            path = self._weight_path()
            state = load_file(str(path), device="cpu")
            model = timm.create_model(f"vit_small_patch16_{self.size}.augreg_in21k_ft_in1k",
                                      pretrained=False, num_classes=1)
            model.load_state_dict({key.removeprefix("vit."): value for key, value in state.items()}, strict=True)
            self._model = model.to(self.device).eval()

    def predict_batch(self, images):
        self._load()
        values = torch.stack([self._transform(image.convert("RGB")) for image in images]).to(self.device)
        with torch.inference_mode():
            scores = self._model(values).flatten().float().sigmoid().cpu().numpy()
        if not np.isfinite(scores).all():
            raise FloatingPointError("Community Forensics produced nonfinite scores")
        return scores

    def detect(self, content):
        if not isinstance(content, Image.Image):
            content = Image.fromarray(np.asarray(content))
        score = float(self.predict_batch([content])[0])
        return DetectionResult(score=score, label="ai" if score >= 0.5 else "human", detector_name=self.name,
                               details={"score_semantics": "classifier_probability", "ai_class": 1,
                                        "model_id": self.model_id, "revision": self.revision,
                                        "crop_size": self.size})
