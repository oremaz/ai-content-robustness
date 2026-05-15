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

    Supports two modes:
    - Zero-shot: Uses pre-computed centroids from common LLM outputs.
    - Few-shot: User provides example texts via setup_support_set().
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
                from transformers import AutoModel  # noqa: F401
                self._available = True
            except ImportError:
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

        with torch.no_grad():
            outputs = self._model(**tokens)
            # Use CLS token or mean pooling depending on model architecture
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding = outputs.pooler_output.cpu().numpy()
            else:
                embedding = outputs.last_hidden_state[:, 0].cpu().numpy()

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

            with torch.no_grad():
                outputs = self._model(**tokens)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output.cpu().numpy()
                else:
                    embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()

            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            all_embeddings.append(embeddings / norms)

        return np.vstack(all_embeddings)

    def _init_default_centroids(self):
        """Initialize centroids using synthetic data from the model itself.

        Generate a small set of characteristic texts to bootstrap the centroids.
        AI text is generated by asking a small LM for typical AI-style completions.
        Human text uses short, varied reference sentences.
        """
        if self._ai_centroid is not None:
            return

        # Characteristic AI-generated text patterns (diverse samples)
        ai_samples = [
            "In conclusion, it is important to note that the implications of this research extend far beyond the immediate scope of our investigation. The findings suggest a nuanced interplay between various factors that warrants further examination.",
            "This comprehensive analysis demonstrates that the proposed methodology achieves state-of-the-art performance across multiple benchmark datasets. Our approach leverages the synergistic combination of advanced techniques to address the fundamental challenges.",
            "Furthermore, it is essential to consider the broader context within which these developments are occurring. The rapid advancement of technology has created unprecedented opportunities for innovation and growth across various sectors.",
            "The experimental results presented in this paper provide compelling evidence that our proposed framework effectively addresses the limitations of existing approaches. Through extensive evaluation, we have demonstrated significant improvements.",
            "It is worth noting that while the current study has certain limitations, the overall trajectory of the research points toward promising directions for future work. The methodology presented here can be readily adapted to various domains.",
            "Additionally, the implementation of this system requires careful consideration of multiple factors including scalability, efficiency, and robustness. Our analysis indicates that the optimal configuration balances these competing objectives.",
        ]

        # Characteristic human-written text patterns (more varied, informal)
        human_samples = [
            "So I tried this new coffee place downtown yesterday and honestly? Best espresso I've had in months. The barista really knew what she was doing.",
            "The meeting went about as well as you'd expect. Dave kept going off on tangents about the budget and we barely got through half the agenda.",
            "I can't believe how fast my kid is growing up. Just yesterday she was learning to walk and now she's asking me about algebra homework.",
            "Rain again today. Third day in a row. I'm starting to think my umbrella and I are going to become permanent companions this spring.",
            "Okay so the trick with sourdough is patience. I mean real patience. Like, don't even look at the dough for the first four hours kind of patience.",
            "My neighbor's dog keeps getting into our yard. Not that I mind really, he's a sweet old thing, but the holes in the flower bed are getting ridiculous.",
        ]

        ai_embeddings = self._embed_batch(ai_samples)
        human_embeddings = self._embed_batch(human_samples)

        self._ai_centroid = ai_embeddings.mean(axis=0)
        self._human_centroid = human_embeddings.mean(axis=0)

        # Normalize centroids
        self._ai_centroid /= np.linalg.norm(self._ai_centroid) + 1e-8
        self._human_centroid /= np.linalg.norm(self._human_centroid) + 1e-8

        logger.info(
            "Initialized default style centroids (AI: %d samples, Human: %d samples)",
            len(ai_samples), len(human_samples),
        )

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

        # Initialize centroids on first use if not set via setup_support_set
        self._init_default_centroids()

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
