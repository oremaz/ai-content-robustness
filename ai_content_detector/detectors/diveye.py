"""The authors' released DivEye/BiScope classifier and matching 83 features."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import zlib

import numpy as np

from .ensemble import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class DivEyeDetector(BaseDetector):
    """Frozen inference using the authors' Hugging Face Space release.

    This combined detector differs from the paper's standalone nine-feature
    DivEye. Feature order, backbone identities and AI probability column match
    app.py at SPACE_REVISION. Both language models share the requested device.
    """

    name = "DivEye + BiScope (authors' release)"
    modality = "text"
    SPACE = "pinyuchen/Diveye_AI_text_detector"
    SPACE_REVISION = "d2961b844705c0aa6f95ded41dc24ba18d2d445f"
    CLASSIFIER_SHA256 = "ee1fce303de8b44777b608645ea1bce8c92c55e2a776d80d5662f744b8b64ac7"
    BACKBONES = {
        "div": ("tiiuae/falcon-7b", "ec89142b67d748a1865ea4451372db8313ada0d8"),
        "bi": ("google/gemma-1.1-2b-it", "d750f5eceb83e978c09e2b3597c2a8784e381022"),
    }
    _RESOURCE_ATTRS = (*BaseDetector._RESOURCE_ATTRS, "_bi_model", "_bi_tokenizer", "_classifier")

    def __init__(self, device="auto", classifier_path=None, model_paths=None):
        self._device_spec = device
        self._classifier_path = classifier_path
        self._model_paths = dict(model_paths or {})
        if set(self._model_paths) - set(self.BACKBONES):
            raise ValueError("DivEye model_paths keys must be 'div' and/or 'bi'")
        self._model = self._tokenizer = None
        self._bi_model = self._bi_tokenizer = None
        self._classifier = None
        self._available = None
        self._availability_error = None
        self._classifier_sha256 = None

    def _source(self, key):
        model, revision = self.BACKBONES[key]
        if key in self._model_paths:
            return str(self._model_paths[key]), {}
        return model, {"revision": revision}

    def _load_classifier(self):
        if self._classifier is not None:
            return
        from xgboost import XGBClassifier

        path = self._classifier_path
        if path is None:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(self.SPACE, "model.json", repo_type="space", revision=self.SPACE_REVISION)
        with Path(path).open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        if self._classifier_path is None and fingerprint != self.CLASSIFIER_SHA256:
            raise ValueError("Authors' DivEye classifier checksum differs from the inspected release")
        classifier = XGBClassifier()
        classifier.load_model(path)
        if classifier.n_features_in_ != 83 or list(classifier.classes_) != [0, 1]:
            raise ValueError("DivEye requires the released 83-feature binary classifier schema")
        self._classifier = classifier
        self._classifier_sha256 = fingerprint

    def is_available(self):
        if self._available is None:
            try:
                from transformers import AutoConfig, AutoTokenizer
                self._load_classifier()
                for key in self.BACKBONES:
                    source, kwargs = self._source(key)
                    AutoConfig.from_pretrained(source, **kwargs)
                    AutoTokenizer.from_pretrained(source, use_fast=False, **kwargs)
                self._available = True
            except Exception as error:
                self._availability_error = str(error)
                self._available = False
                logger.warning("DivEye release unavailable: %s", error)
        return bool(self._available)

    def _load(self):
        if self._model is not None and self._bi_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._load_classifier()
        device = self._device_spec
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        for key, model_attr, tokenizer_attr in (
            ("div", "_model", "_tokenizer"), ("bi", "_bi_model", "_bi_tokenizer"),
        ):
            if getattr(self, model_attr) is not None:
                continue
            source, kwargs = self._source(key)
            tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False, **kwargs)
            model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=self._dtype, **kwargs)
            model.to(self._device).eval().requires_grad_(False)
            setattr(self, tokenizer_attr, tokenizer)
            setattr(self, model_attr, model)
        self._available = True

    @staticmethod
    def _div_features(surprisals, text):
        from scipy.stats import entropy, kurtosis, skew

        values = np.asarray(surprisals)
        if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
            raise ValueError("DivEye requires at least four finite token surprisals")
        first = np.diff(values)
        second = np.diff(np.diff(-values))
        constant = np.var(values) == 0
        autocorrelation = (np.corrcoef(second[:-1], second[1:])[0, 1]
                           if len(second) > 1 and np.std(second[:-1]) and np.std(second[1:])
                           else np.nan)
        encoded = text.encode("utf-8")
        return np.asarray([
            np.mean(values), np.std(values), np.var(values),
            np.nan if constant else skew(values), np.nan if constant else kurtosis(values),
            np.mean(first), np.std(first), np.var(second),
            entropy(np.histogram(second, bins=20, density=True)[0]), autocorrelation,
            len(zlib.compress(encoded)) / len(encoded),
        ], dtype=np.float64)

    @staticmethod
    def _biscope_features(forward_losses, backward_losses):
        features = []
        if len(forward_losses) != len(backward_losses) or not len(forward_losses):
            raise ValueError("BiScope losses must be nonempty and aligned")
        for decile in range(1, 10):
            split = len(forward_losses) * decile // 10
            for values in (forward_losses, backward_losses):
                clipped = np.nan_to_num(np.clip(values[split:], -1e6, 1e6),
                                        nan=0.0, posinf=1e6, neginf=-1e6)
                features.extend([np.mean(clipped), np.max(clipped), np.min(clipped), np.std(clipped)])
        return np.asarray(features, dtype=np.float64)

    def extract_features(self, content):
        import torch
        import torch.nn.functional as functional

        text = content.strip()
        if len(text.split()) < 15:
            raise ValueError("The released DivEye detector requires at least 15 words")
        self._load()
        with torch.inference_mode():
            tokens = self._tokenizer.encode(text, return_tensors="pt", truncation=True,
                                            max_length=1024).to(self._device)
            logits = self._model(tokens, use_cache=False).logits[0, :-1].float()
            targets = tokens[0, 1:]
            log_probabilities = torch.log_softmax(logits, dim=-1)
            surprisals = -log_probabilities.gather(1, targets[:, None]).squeeze(1).cpu().numpy()
            div_features = self._div_features(surprisals, text)
            del logits, log_probabilities, tokens, targets

            prompt = self._bi_tokenizer("Complete the following text: ", return_tensors="pt").input_ids
            text_ids = self._bi_tokenizer(text, return_tensors="pt", max_length=2000,
                                          truncation=True).input_ids
            combined = torch.cat([prompt, text_ids], dim=1).to(self._device)
            start, end = prompt.shape[1], combined.shape[1]
            logits = self._bi_model(input_ids=combined, use_cache=False).logits[0]
            targets = combined[0, start:end]
            forward = functional.cross_entropy(logits[start - 1:end - 1], targets, reduction="none").cpu().numpy()
            backward = functional.cross_entropy(logits[start:end], targets, reduction="none").cpu().numpy()
        features = np.concatenate([div_features, self._biscope_features(forward, backward)])
        # The released XGBoost model accepts NaN moments as missing features.
        if features.shape != (83,) or np.isinf(features).any():
            raise ValueError("Invalid released DivEye feature vector")
        return features

    def detect(self, content):
        features = self.extract_features(content)
        probabilities = self._classifier.predict_proba(features.reshape(1, -1))
        ai_score = float(probabilities[0, 1])
        if not np.isfinite(ai_score) or not 0 <= ai_score <= 1:
            raise ValueError("DivEye returned an invalid AI probability")
        return DetectionResult(
            score=ai_score, label=DetectionResult.label_from_score(ai_score), detector_name=self.name,
            details={"variant": "authors_diveye_biscope_83_features",
                     "space_revision": self.SPACE_REVISION,
                     "classifier_sha256": self._classifier_sha256,
                     "classifier_origin": "authors_release" if self._classifier_path is None else "supplied_checkpoint",
                     "backbones": {key: {"model": model, "revision": revision}
                                   for key, (model, revision) in self.BACKBONES.items()},
                     "local_model_paths": self._model_paths,
                     "score_semantics": "AI probability is column 1", "feature_count": 83,
                     "frozen": True},
        )
