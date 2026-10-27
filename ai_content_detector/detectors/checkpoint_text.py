"""Checkpoint-backed text classifiers with AI-oriented scores."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .ensemble import BaseDetector, DetectionResult

logger = logging.getLogger("ai_content_detector.detectors.text_detectors")


class ModernBERTDetector(BaseDetector):
    """QLoRA-finetuned ModernBERT from bench-aitextdetect."""

    __module__ = "ai_content_detector.detectors.text_detectors"

    name = "ModernBERT (QLoRA)"
    modality = "text"

    def __init__(
        self,
        model_name: str = "qlora_modernbert_base",
        path_start: str = "bench_aitextdetect",
        device: str = "auto",
    ):
        self._model_name = model_name
        self._path_start = path_start
        self._device = device
        self._model = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from ml_pipeline.pipelines_torch.models import HuggingFaceQLoRAWrapper
            from ml_pipeline.pipelines_torch.vision_models import MODEL_REGISTRY  # noqa: F401
            from ml_pipeline.utils.utils import load_model_by_name

            device = self._device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = load_model_by_name(
                HuggingFaceQLoRAWrapper,
                self._model_name,
                {"num_labels": 2, "device": device},
                path_start=self._path_start,
            )
            self._model.eval()
            self._available = True
        except Exception as e:
            logger.warning("ModernBERTDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            self._load()
        return bool(self._available)

    def detect(self, content: str) -> DetectionResult:
        self._load()
        if not self._available:
            raise RuntimeError("ModernBERT checkpoint not available")

        import torch

        tokens = self._model.tokenizer(
            content,
            return_tensors="pt",
            truncation=True,
            max_length=self._model.max_seq_length,
            padding=True,
        )
        tokens = {k: v.to(self._model.device) for k, v in tokens.items()}

        with torch.no_grad():
            logits = self._model.model(**tokens).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # MAGE convention: label 0 = machine, label 1 = human
        ai_score = float(probs[0])
        return DetectionResult(
            score=ai_score,
            label=DetectionResult.label_from_score(ai_score),
            details={"prob_ai": float(probs[0]), "prob_human": float(probs[1])},
        )


class TFIDFDetector(BaseDetector):
    """TF-IDF + sklearn classifier from bench-aitextdetect."""

    __module__ = "ai_content_detector.detectors.text_detectors"

    name = "TF-IDF + LogReg"
    modality = "text"

    def __init__(
        self,
        model_name: str = "tfidf_logreg",
        path_start: str = "bench_aitextdetect",
    ):
        self._model_name = model_name
        self._path_start = path_start
        self._model = None
        self._vectorizer = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import joblib
            from sklearn.linear_model import LogisticRegression
            from ml_pipeline.utils.utils import load_model_by_name
            from ml_pipeline.pipelines_torch.models import SklearnRandomForestClassifierWrapper

            # Determine model class from name
            if "random_forest" in self._model_name:
                model_cls = SklearnRandomForestClassifierWrapper
            else:
                model_cls = lambda **kw: LogisticRegression(max_iter=1000)

            self._model = load_model_by_name(
                model_cls,
                self._model_name,
                {"input_dim": 50000, "num_classes": 2},
                path_start=self._path_start,
            )

            # Try to load saved vectorizer
            from ml_pipeline.utils.utils import RESULTS_DIR_IN
            vec_path = Path(RESULTS_DIR_IN) / self._path_start / "tfidf_vectorizer.pkl"
            if vec_path.exists():
                self._vectorizer = joblib.load(vec_path)
            else:
                # Fallback: create a new one (will need fitting: mark unavailable for now)
                logger.warning("TF-IDF vectorizer not found at %s", vec_path)
                self._available = False
                return

            self._available = True
        except Exception as e:
            logger.warning("TFIDFDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            self._load()
        return bool(self._available)

    def detect(self, content: str) -> DetectionResult:
        self._load()
        if not self._available:
            raise RuntimeError("TF-IDF model/vectorizer not available")

        features = self._vectorizer.transform([content]).toarray().astype(np.float32)
        probs = self._model.predict_proba(features)[0]

        ai_score = float(probs[0])
        return DetectionResult(
            score=ai_score,
            label=DetectionResult.label_from_score(ai_score),
            details={"prob_ai": float(probs[0]), "prob_human": float(probs[1])},
        )

