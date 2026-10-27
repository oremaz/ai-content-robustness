"""Paraphrase and disruption-recovery text detectors."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from .ensemble import BaseDetector, DetectionResult
import os
import re

logger = logging.getLogger("ai_content_detector.detectors.text_detectors")


class ParaphraseRoundTripDetector(BaseDetector):
    """Round-trip paraphrasing defense (DIPPER-inspired, Krishna et al. NeurIPS 2023).

    If a text requires heavy rewriting to be expressed in standard prose, it is
    either adversarially optimized or otherwise far from the natural-language
    manifold. We ask a rewriter model to normalize the input and measure the
    edit distance between input and rewrite; large distance → suspected AI.

    This is *not* a reproduction of a specific NeurIPS 2025 "Paraphrase
    Inversion" paper (no such paper was located in our literature search: the
    original naming conflated several lines of work). It is a lightweight
    round-trip heuristic, useful as an auxiliary ensemble member but not a
    stand-alone SOTA detector.
    """

    __module__ = "ai_content_detector.detectors.text_detectors"

    name = "Paraphrase Round-Trip"
    modality = "text"

    def __init__(self, model_name: str = "Qwen/Qwen3.5-4B", device: str = "auto"):
        self._model_name = model_name
        self._device_spec = device
        self._pipe = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from transformers import pipeline

            device_idx = 0 if (self._device_spec == "auto" and torch.cuda.is_available()) else -1
            if self._device_spec not in ("auto", "cpu", "cuda"):
                device_idx = int(self._device_spec.split(":")[-1]) if ":" in self._device_spec else -1

            self._pipe = pipeline(
                "text-generation",
                model=self._model_name,
                device=device_idx,
                torch_dtype=torch.float16 if device_idx >= 0 else torch.float32,
            )
            self._available = True
        except Exception as e:
            logger.warning("ParaphraseRoundTripDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from transformers import pipeline  # noqa: F401
                import editdistance  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return bool(self._available)

    def detect(self, content: Union[str, List[str]]) -> Union[DetectionResult, List[DetectionResult]]:
        self._load()
        if not self._available:
            raise RuntimeError("ParaphraseRoundTripDetector model not loaded")

        is_single = isinstance(content, str)
        texts = [content] if is_single else content

        import editdistance
        results = []
        for text in texts:
            if len(text.strip()) < 20:
                results.append(DetectionResult(score=0.5, label="uncertain", details={"reason": "too short"}))
                continue

            prompt = (
                "Rewrite the following text into clear, standard, and predictable "
                "prose without stylistic flourishes:\n\n"
                f"{text}\n\nRewritten text:"
            )
            try:
                out = self._pipe(
                    prompt, max_new_tokens=200, return_full_text=False,
                )[0]["generated_text"].strip()

                dist = editdistance.eval(text.split(), out.split())
                max_len = max(len(text.split()), len(out.split()), 1)
                normalized_dist = dist / max_len
                # Large distance → the original was far from clear prose; score as AI.
                ai_score = min(normalized_dist * 1.5, 1.0)
                ai_score = self._complexity_penalty(text, ai_score)

                results.append(DetectionResult(
                    score=ai_score,
                    label=DetectionResult.label_from_score(ai_score),
                    details={"edit_distance": dist, "normalized_distance": normalized_dist, "rewritten": out},
                ))
            except Exception as e:
                logger.error(f"ParaphraseRoundTripDetector failed: {e}")
                results.append(DetectionResult(score=0.5, label="error", details={"error": str(e)}))

        return results[0] if is_single else results


class DisruptRecoverDetector(BaseDetector):
    """D&R-style black-box detector via local disruption and single recovery call.

    Sun et al. (ICLR 2026) propose Disrupt-and-Recover (D&R): corrupt the input
    locally, ask a black-box LLM to recover it once, and use posterior
    concentration as the signal. AI text tends to be recovered more exactly
    than human text after the same corruption.

    Word-chunk shuffling, constrained recovery, BERTScore, and word-order rank
    correlations follow the paper protocol.
    """

    __module__ = "ai_content_detector.detectors.text_detectors"

    name = "Disrupt-and-Recover"
    modality = "text"

    def __init__(
        self,
        recover_fn: Optional[Callable[[str], str]] = None,
        semantic_scorer: Optional[Callable[[str, str], float]] = None,
        min_words: int = 35,
        threshold: float = 0.72,
        temperature: float = 0.08,
        seed: int = 13,
        api_model: Optional[str] = None,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self._recover_fn = recover_fn
        self._semantic_scorer = semantic_scorer
        self._min_words = int(min_words)
        self._threshold = float(threshold)
        self._temperature = float(temperature)
        self._seed = int(seed)
        self._api_model = api_model or os.environ.get("DR_RECOVERY_MODEL")
        self._api_base_url = api_base_url or os.environ.get("OPENAI_BASE_URL")
        if self._api_base_url is None and os.environ.get("OPENROUTER_API_KEY"):
            self._api_base_url = "https://openrouter.ai/api/v1"
        self._api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
        )
        self._client = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._recover_fn is not None:
            return True
        if self._available is None:
            try:
                from openai import OpenAI  # noqa: F401
                self._available = bool(self._api_key and self._api_model)
            except ImportError:
                self._available = False
        return bool(self._available)

    @staticmethod
    def _word_tokens(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)

    def _disrupt(self, text: str) -> str:
        import random
        return self._shuffle_within_chunks(text, random.Random(self._seed))

    def _shuffle_within_chunks(self, text: str, rng: "random.Random") -> str:
        chunks = re.split(r"([.!?;:\n]+)", text)
        out: List[str] = []
        for chunk in chunks:
            if re.fullmatch(r"[.!?;:\n]+", chunk or ""):
                out.append(chunk)
                continue

            parts = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|\s+|[^\w\s]", chunk)
            word_positions = [
                i for i, part in enumerate(parts)
                if re.fullmatch(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", part)
            ]
            if len(word_positions) < 3:
                out.append(chunk)
                continue

            selected = word_positions
            words = [parts[index] for index in selected]
            rng.shuffle(words)
            if words == [parts[i] for i in selected] and len(words) > 1:
                words = words[1:] + words[:1]
            for idx, word in zip(selected, words):
                parts[idx] = word
            out.append("".join(parts))
        return "".join(out)

    @staticmethod
    def _format_recovery_prompt(corrupted_text: str) -> str:
        return (
            "Recover the original passage from the corrupted version below. "
            "Words were reordered only within punctuation-delimited chunks. "
            "Do not add or remove any words. Restore only their original order. "
            "Return only the recovered passage, with no explanation.\n\n"
            f"Corrupted passage:\n{corrupted_text}\n\nRecovered passage:"
        )

    def _load_client(self):
        if self._client is not None:
            return
        from openai import OpenAI

        kwargs: Dict[str, str] = {"api_key": self._api_key or ""}
        if self._api_base_url:
            kwargs["base_url"] = self._api_base_url
        self._client = OpenAI(**kwargs)

    def _recover(self, corrupted_text: str) -> str:
        if self._recover_fn is not None:
            return self._recover_fn(corrupted_text)
        if not self.is_available():
            raise RuntimeError(
                "DisruptRecoverDetector needs recover_fn or an OpenAI-compatible "
                "API key plus DR_RECOVERY_MODEL/api_model."
            )
        self._load_client()
        prompt = self._format_recovery_prompt(corrupted_text)
        response = self._client.chat.completions.create(
            model=self._api_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    def _semantic_similarity(self, source: str, recovered: str) -> float:
        if self._semantic_scorer is not None:
            return float(self._semantic_scorer(source, recovered))
        try:
            from bert_score import score as bert_score
        except ImportError as error:
            raise RuntimeError(
                "D&R semantic scoring requires the bert-score package or semantic_scorer"
            ) from error
        _, _, f1 = bert_score([recovered], [source], lang="en", verbose=False)
        return float(f1[0].item())

    @staticmethod
    def _structural_similarity(source: str, recovered: str) -> float:
        from collections import defaultdict, deque
        from scipy.stats import kendalltau, spearmanr

        source_words = DisruptRecoverDetector._word_tokens(source.lower())
        recovered_words = DisruptRecoverDetector._word_tokens(recovered.lower())
        positions = defaultdict(deque)
        for index, word in enumerate(recovered_words):
            positions[word].append(index)
        recovered_ranks = []
        source_ranks = []
        for index, word in enumerate(source_words):
            if positions[word]:
                source_ranks.append(index)
                recovered_ranks.append(positions[word].popleft())
        if len(source_ranks) < 2:
            return 0.0
        coverage = len(source_ranks) / max(len(source_words), len(recovered_words), 1)
        kendall = float(np.nan_to_num(kendalltau(source_ranks, recovered_ranks).statistic))
        spearman = float(np.nan_to_num(spearmanr(source_ranks, recovered_ranks).statistic))
        correlation = ((kendall + spearman) / 2.0 + 1.0) / 2.0
        return float(np.clip(coverage * correlation, 0.0, 1.0))

    def _similarity_to_ai_score(self, similarity: float) -> float:
        import math

        temp = max(self._temperature, 1e-6)
        score = 1.0 / (1.0 + math.exp(-(similarity - self._threshold) / temp))
        return float(max(0.0, min(1.0, score)))

    def detect(self, content: Union[str, List[str]]) -> Union[DetectionResult, List[DetectionResult]]:
        if not self.is_available():
            raise RuntimeError("Disrupt-and-Recover recovery model is not configured")

        is_single = isinstance(content, str)
        texts = [content] if is_single else content

        results = []
        for text in texts:
            word_count = len(self._word_tokens(text))
            if word_count < self._min_words:
                results.append(DetectionResult(
                    score=0.5,
                    label="uncertain",
                    detector_name=self.name,
                    details={
                        "reason": "too short for stable disrupt-and-recover scoring",
                        "word_count": word_count,
                        "min_words": self._min_words,
                    },
                ))
                continue

            try:
                corrupted = self._disrupt(text)
                recovered = self._recover(corrupted).strip()
                semantic_similarity = self._semantic_similarity(text, recovered)
                structural_similarity = self._structural_similarity(text, recovered)
                similarity = 0.5 * semantic_similarity + 0.5 * structural_similarity
                ai_score = self._similarity_to_ai_score(similarity)
                results.append(DetectionResult(
                    score=ai_score,
                    label=DetectionResult.label_from_score(ai_score),
                    detector_name=self.name,
                    details={
                        "recovery_similarity": similarity,
                        "semantic_bertscore": semantic_similarity,
                        "structural_rank_similarity": structural_similarity,
                        "corrupted": corrupted,
                        "recovered": recovered,
                        "threshold": self._threshold,
                        "disruption_mode": "word_chunk_shuffle",
                    },
                ))
            except Exception as e:
                logger.error("DisruptRecoverDetector failed: %s", e)
                results.append(DetectionResult(
                    score=0.5,
                    label="error",
                    detector_name=self.name,
                    details={"error": str(e)},
                ))

        return results[0] if is_single else results


InversionDetector = ParaphraseRoundTripDetector
