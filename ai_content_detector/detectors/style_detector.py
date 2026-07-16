"""Style embedding-based AI text detector.

Implements few-shot AI text detection using learned writing style
representations, following Rivera Soto et al. (2024), "Few-Shot Detection
of Machine-Generated Text using Style Representations" (arXiv:2401.06712).

Core idea: LLMs have consistent writing styles that differ from humans in
a learned style embedding space. A RoBERTa model trained with supervised
contrastive learning (LUAR — Linguistically-Informed Universal Authorship
Representation) maps text to style vectors where cosine similarity reflects
stylistic similarity. Detection is few-shot: cosine similarity between the
query embedding and a centroid of known LLM-generated text.

This detector uses orthogonal features from the other detectors in the zoo
(perplexity, frequency artifacts, etc.), testing whether RL evasion can
learn to shift writing style — a stronger test of the evasion framework.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .ensemble import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class StyleEmbeddingDetector(BaseDetector):
    """Few-shot AI text detector using learned style representations.

    Uses LUAR (rrivera1849/LUAR-MUD), a RoBERTa-based model trained with
    supervised contrastive learning on 5M+ Reddit authors, to embed text
    into a style space. Detection works by comparing query embeddings to
    pre-computed centroids of human vs. LLM-generated text.

    A labeled support set must be supplied with ``setup_support_set``. The
    detector does not fabricate a default support distribution.
    """

    name = "Style Embedding"
    modality = "text"

    def __init__(
        self,
        model_name: str = "rrivera1849/LUAR-MUD",
        max_length: int = 512,
        device: str = "auto",
    ):
        self._model_name = model_name
        self._max_length = max_length
        self._device_spec = device
        self._model = None
        self._tokenizer = None
        self._available: Optional[bool] = None
        # Centroids for detection (populated by _init_default_centroids or setup_support_set)
        self._ai_centroid: Optional[np.ndarray] = None
        self._human_centroid: Optional[np.ndarray] = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer

            device = self._device_spec
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device

            # LUAR ships custom modeling code on the Hub, so trust_remote_code
            # is required — without it AutoModel.from_pretrained raises.
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True,
            )
            self._model = AutoModel.from_pretrained(
                self._model_name, trust_remote_code=True,
            )
            self._model.to(device)
            self._model.eval()

            self._available = True
            logger.info("StyleEmbeddingDetector loaded: %s on %s", self._model_name, device)
        except Exception as e:
            logger.warning("StyleEmbeddingDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from transformers import AutoConfig
                AutoConfig.from_pretrained(self._model_name, trust_remote_code=True)
                self._available = True
            except Exception as error:
                self._availability_error = str(error)
                self._available = False
        return bool(self._available)

    def _embed_single(self, text: str) -> np.ndarray:
        """Embed a single text into the style space."""
        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=True,
        ).to(self._device)
        tokens = {name: tensor.unsqueeze(1) for name, tensor in tokens.items()}

        with torch.no_grad():
            embedding = self._model(**tokens).cpu().numpy()

        # L2 normalize
        norm = np.linalg.norm(embedding) + 1e-8
        return (embedding / norm).flatten()

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts into the style space."""
        all_embeddings = []
        # Process in mini-batches for memory efficiency
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
                padding=True,
            ).to(self._device)
            tokens = {name: tensor.unsqueeze(1) for name, tensor in tokens.items()}

            with torch.no_grad():
                embeddings = self._model(**tokens).cpu().numpy()

            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            all_embeddings.append(embeddings / norms)

        return np.vstack(all_embeddings)

    def setup_support_set(
        self,
        ai_texts: List[str],
        human_texts: List[str],
    ):
        """Set up few-shot support set with labeled examples.

        Args:
            ai_texts: Known AI-generated texts (support set for AI class).
            human_texts: Known human-written texts (support set for human class).
        """
        self._load()
        if not self._available:
            raise RuntimeError("Style embedding model not available")
        if not ai_texts or not human_texts:
            raise ValueError("Both AI and human support sets must be nonempty")

        ai_embeddings = self._embed_batch(ai_texts)
        human_embeddings = self._embed_batch(human_texts)

        self._ai_centroid = ai_embeddings.mean(axis=0)
        self._human_centroid = human_embeddings.mean(axis=0)

        self._ai_centroid /= np.linalg.norm(self._ai_centroid) + 1e-8
        self._human_centroid /= np.linalg.norm(self._human_centroid) + 1e-8

        logger.info(
            "Style support set configured: %d AI texts, %d human texts",
            len(ai_texts), len(human_texts),
        )

    def get_style_embedding(self, text: str) -> np.ndarray:
        """Get the style embedding for a text (useful for MultiSPIN integration)."""
        self._load()
        if not self._available:
            raise RuntimeError("Style embedding model not available")
        return self._embed_single(text)

    def get_style_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get style embeddings for a batch of texts."""
        self._load()
        if not self._available:
            raise RuntimeError("Style embedding model not available")
        return self._embed_batch(texts)

    def detect(self, content: str) -> DetectionResult:
        """Detect AI-generated text using style embedding similarity.

        Computes cosine similarity of the query embedding to both the AI
        and human centroids. The AI score is derived from the relative
        similarity: closer to AI centroid = higher AI score.
        """
        self._load()
        if not self._available:
            raise RuntimeError("Style embedding model not available")

        if self._ai_centroid is None or self._human_centroid is None:
            raise RuntimeError(
                "StyleEmbeddingDetector requires setup_support_set() before detection"
            )

        query_embedding = self._embed_single(content)

        # Cosine similarity to each centroid (embeddings are already L2-normalized)
        sim_ai = float(np.dot(query_embedding, self._ai_centroid))
        sim_human = float(np.dot(query_embedding, self._human_centroid))

        # Convert to AI probability via softmax over similarities
        # Temperature controls sharpness of the decision boundary
        temperature = 0.5
        exp_ai = np.exp(sim_ai / temperature)
        exp_human = np.exp(sim_human / temperature)
        ai_score = exp_ai / (exp_ai + exp_human)

        return DetectionResult(
            score=float(ai_score),
            label=DetectionResult.label_from_score(float(ai_score)),
            details={
                "sim_ai_centroid": sim_ai,
                "sim_human_centroid": sim_human,
                "style_embedding_dim": len(query_embedding),
            },
        )
