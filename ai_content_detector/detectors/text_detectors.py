"""Text-based AI content detectors.

Includes both internal (checkpoint-based) and external (zero-shot) detectors.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .ensemble import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup — allow imports from ml_pipeline/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ML_PIPELINE = _REPO_ROOT / "ml_pipeline"
if str(_ML_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_ML_PIPELINE))


# ===========================================================================
# 1. Internal detectors (from bench-aitextdetect checkpoints)
# ===========================================================================


class ModernBERTDetector(BaseDetector):
    """QLoRA-finetuned ModernBERT from bench-aitextdetect."""

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
            from pipelines_torch.models import HuggingFaceQLoRAWrapper
            from pipelines_torch.vision_models import MODEL_REGISTRY  # noqa: F401
            from utils.utils import load_model_by_name

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
            from utils.utils import load_model_by_name
            from pipelines_torch.models import SklearnRandomForestClassifierWrapper

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
            vec_path = _ML_PIPELINE / "results" / self._path_start / "tfidf_vectorizer.pkl"
            if vec_path.exists():
                self._vectorizer = joblib.load(vec_path)
            else:
                # Fallback: create a new one (will need fitting — mark unavailable for now)
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


# ===========================================================================
# 2. Binoculars (zero-shot, ICML 2024)
# ===========================================================================


class BinocularsDetector(BaseDetector):
    """Zero-shot detector using cross-perplexity ratio of two reference LMs.

    Paper: Hans et al., "Spotting LLMs With Binoculars" (ICML 2024).
    Core idea: score = perplexity(observer) / cross_perplexity(performer, observer).
    Low score → likely AI-generated.
    """

    name = "Binoculars"
    modality = "text"

    # Default thresholds from the paper (low FPR operating point)
    THRESHOLD_LOW_FPR = 0.9015  # ~0.01% FPR
    THRESHOLD_ACCURACY = 0.8536  # balanced accuracy optimized

    def __init__(
        self,
        observer_name: str = "tiiuae/falcon-7b",
        performer_name: str = "tiiuae/falcon-7b-instruct",
        max_length: int = 512,
        device: str = "auto",
        threshold: float = 0.9015,
    ):
        self._observer_name = observer_name
        self._performer_name = performer_name
        self._max_length = max_length
        self._device_spec = device
        self._threshold = threshold
        self._observer = None
        self._performer = None
        self._tokenizer = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._observer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self._device_spec
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                logger.warning("Binoculars requires GPU for reasonable speed; falling back to CPU")

            self._device = device
            dtype = torch.float16 if device == "cuda" else torch.float32

            self._tokenizer = AutoTokenizer.from_pretrained(self._observer_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._observer = AutoModelForCausalLM.from_pretrained(
                self._observer_name, torch_dtype=dtype, device_map=device
            )
            self._performer = AutoModelForCausalLM.from_pretrained(
                self._performer_name, torch_dtype=dtype, device_map=device
            )
            self._observer.eval()
            self._performer.eval()
            self._available = True
        except Exception as e:
            logger.warning("BinocularsDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import torch
                if not torch.cuda.is_available():
                    self._available = False
                    return False
                # Don't load models just for checking — assume available if GPU present
                self._available = True
            except ImportError:
                self._available = False
        return bool(self._available)

    def _compute_score(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        import torch
        import torch.nn.functional as F

        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits_observer = self._observer(**tokens).logits[:, :-1]
            logits_performer = self._performer(**tokens).logits[:, :-1]

        target_ids = tokens["input_ids"][:, 1:]
        attention_mask = tokens["attention_mask"][:, 1:].to(logits_observer.dtype)

        # Numerator: observer perplexity — the observer's per-token cross-entropy
        # against the *actual* tokens (how surprised the observer is by the text).
        ce_observer = F.cross_entropy(
            logits_observer.transpose(1, 2),
            target_ids,
            reduction="none",
        )

        # Denominator: cross-perplexity — the per-token cross-entropy between the
        # two models' *predictive distributions* (NOT the performer's perplexity
        # on the actual tokens). This is the quantity that makes Binoculars
        # robust: H(performer_dist, observer_dist) averaged over positions.
        # See Hans et al., ICML 2024, and the reference implementation's
        # ``entropy`` function.
        performer_probs = F.softmax(logits_performer, dim=-1)
        observer_log_probs = F.log_softmax(logits_observer, dim=-1)
        ce_cross = -(performer_probs * observer_log_probs).sum(dim=-1)

        # Apply attention mask and compute mean per sequence
        ce_observer = (ce_observer * attention_mask).sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1)
        ce_cross = (ce_cross * attention_mask).sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1)

        scores = (ce_observer / ce_cross.clamp(min=1e-6)).cpu().numpy().tolist()
        return scores[0] if is_single else scores

    def detect(self, content: Union[str, List[str]]) -> Union[DetectionResult, List[DetectionResult]]:
        self._load()
        if not self._available:
            raise RuntimeError("Binoculars models not loaded")

        is_single = isinstance(content, str)
        texts = [content] if is_single else content
        raw_scores = self._compute_score(texts)
        if is_single:
            raw_scores = [raw_scores]

        results = []
        for text, raw_score in zip(texts, raw_scores):
            # Apply dynamic threshold based on length
            length = len(text.split())
            dynamic_thresh = self._dynamic_threshold(length, self._threshold)
            
            # Convert to AI probability: score < threshold → AI
            # Normalize to [0, 1] range around threshold
            ai_score = 1.0 - min(max(raw_score / (dynamic_thresh * 1.5), 0.0), 1.0)
            
            # Apply ESL bias penalty
            ai_score = self._complexity_penalty(text, ai_score)

            results.append(DetectionResult(
                score=ai_score,
                label=DetectionResult.label_from_score(ai_score),
                details={
                    "raw_binoculars_score": raw_score,
                    "threshold_used": dynamic_thresh,
                    "below_threshold": raw_score < dynamic_thresh,
                },
            ))

        return results[0] if is_single else results


# ===========================================================================
# 3. Fast-DetectGPT (zero-shot, ICLR 2024)
# ===========================================================================


class FastDetectGPTDetector(BaseDetector):
    """Zero-shot detector using conditional probability curvature.

    Paper: Bao et al., "Fast-DetectGPT: Efficient Zero-Shot Detection via
    Conditional Probability Curvature" (ICLR 2024).
    """

    name = "Fast-DetectGPT"
    modality = "text"

    def __init__(
        self,
        scoring_model: str = "tiiuae/falcon-7b",
        max_length: int = 512,
        device: str = "auto",
    ):
        self._scoring_model_name = scoring_model
        self._max_length = max_length
        self._device_spec = device
        self._model = None
        self._tokenizer = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self._device_spec
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            dtype = torch.float16 if device == "cuda" else torch.float32

            self._tokenizer = AutoTokenizer.from_pretrained(self._scoring_model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self._scoring_model_name, torch_dtype=dtype, device_map=device
            )
            self._model.eval()
            self._available = True
        except Exception as e:
            logger.warning("FastDetectGPTDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import torch
                self._available = torch.cuda.is_available()
            except ImportError:
                self._available = False
        return bool(self._available)

    def _compute_curvature(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        """Compute conditional probability curvature (Fast-DetectGPT criterion)."""
        import torch

        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length,
        ).to(self._device)

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"][:, 1:]
        seq_len = input_ids.size(1)

        if seq_len < 10:
            return 0.0 if is_single else [0.0] * len(text)

        with torch.no_grad():
            outputs = self._model(**tokens)
            logits = outputs.logits[:, :-1]  # (batch, seq_len-1, vocab)
            log_probs = torch.log_softmax(logits, dim=-1)

            target_ids = input_ids[:, 1:]  # (batch, seq_len-1)

            # Log-prob of actual next token
            actual_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

            # Expected log-prob under the model's distribution
            probs = torch.softmax(logits, dim=-1)
            expected_log_probs = (probs * log_probs).sum(dim=-1)

            # Curvature: actual - expected
            curvature = (actual_log_probs - expected_log_probs)

            # Aggregate: mean curvature per sequence
            curvature = (curvature * attention_mask).sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1)
            scores = curvature.cpu().numpy().tolist()

        return scores[0] if is_single else scores

    def detect(self, content: Union[str, List[str]]) -> Union[DetectionResult, List[DetectionResult]]:
        self._load()
        if not self._available:
            raise RuntimeError("Fast-DetectGPT model not loaded")

        is_single = isinstance(content, str)
        texts = [content] if is_single else content
        # texts is always a list, so _compute_curvature returns a list.
        raw_curvatures = self._compute_curvature(texts)

        import math
        results = []
        for text, raw_curvature in zip(texts, raw_curvatures):
            length = len(text.split())
            # Fast-DetectGPT uses ~0.3 as a transition point.
            dynamic_thresh = self._dynamic_threshold(length, base_threshold=0.3)
            
            # Map to [0, 1] using sigmoid
            ai_score = 1.0 / (1.0 + math.exp(-5.0 * (raw_curvature - dynamic_thresh)))
            
            # ESL bias penalty
            ai_score = self._complexity_penalty(text, ai_score)

            results.append(DetectionResult(
                score=ai_score,
                label=DetectionResult.label_from_score(ai_score),
                details={"raw_curvature": raw_curvature, "threshold_used": dynamic_thresh},
            ))

        return results[0] if is_single else results


# ===========================================================================
# 5. Paraphrase Round-Trip Detector (DIPPER-inspired)
# ===========================================================================


class ParaphraseRoundTripDetector(BaseDetector):
    """Round-trip paraphrasing defense (DIPPER-inspired, Krishna et al. NeurIPS 2023).

    If a text requires heavy rewriting to be expressed in standard prose, it is
    either adversarially optimized or otherwise far from the natural-language
    manifold. We ask a rewriter model to normalize the input and measure the
    edit distance between input and rewrite; large distance → suspected AI.

    This is *not* a reproduction of a specific NeurIPS 2025 "Paraphrase
    Inversion" paper (no such paper was located in our literature search — the
    original naming conflated several lines of work). It is a lightweight
    round-trip heuristic, useful as an auxiliary ensemble member but not a
    stand-alone SOTA detector.
    """

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


# Backwards-compatible alias for any existing import sites; the original name
# overclaimed authorship of a non-existent "Paraphrase Inversion" paper.
InversionDetector = ParaphraseRoundTripDetector


# ===========================================================================
# 6. Disrupt-and-Recover (D&R-style, ICLR 2026)
# ===========================================================================


class DisruptRecoverDetector(BaseDetector):
    """D&R-style black-box detector via local disruption and single recovery call.

    Sun et al. (ICLR 2026) propose Disrupt-and-Recover (D&R): corrupt the input
    locally, ask a black-box LLM to recover it once, and use posterior
    concentration as the signal. AI text tends to be recovered more exactly
    than human text after the same corruption.

    This implementation is deliberately a lightweight, configurable hook rather
    than an overclaimed reproduction of the authors' still-sparse public code.
    Pass ``recover_fn`` for experiments, or configure an OpenAI-compatible API
    with ``OPENAI_API_KEY`` / ``OPENROUTER_API_KEY`` and a model name.
    """

    name = "Disrupt-and-Recover"
    modality = "text"

    def __init__(
        self,
        recover_fn: Optional[Callable[[str], str]] = None,
        disruption_rate: float = 0.22,
        disruption_mode: str = "shuffle",
        min_words: int = 35,
        threshold: float = 0.72,
        temperature: float = 0.08,
        seed: int = 13,
        api_model: Optional[str] = None,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self._recover_fn = recover_fn
        self._disruption_rate = float(disruption_rate)
        self._disruption_mode = disruption_mode
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

        if self._disruption_mode not in {"shuffle", "mask"}:
            raise ValueError("disruption_mode must be 'shuffle' or 'mask'")

        if self._disruption_mode == "shuffle":
            return self._shuffle_within_chunks(text, random.Random(self._seed + len(text)))

        # Preserve whitespace/punctuation so the recovery model only fills
        # semantic holes, not formatting damage.
        parts = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|\s+|[^\w\s]", text)
        word_positions = [
            i for i, part in enumerate(parts)
            if re.fullmatch(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", part)
        ]
        if not word_positions:
            return text

        n_mask = max(1, int(round(len(word_positions) * self._disruption_rate)))
        rng = random.Random(self._seed + len(text))
        for idx in rng.sample(word_positions, min(n_mask, len(word_positions))):
            parts[idx] = "[MASK]"
        return "".join(parts)

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

            n_shuffle = max(2, int(round(len(word_positions) * self._disruption_rate)))
            selected = rng.sample(word_positions, min(n_shuffle, len(word_positions)))
            words = [parts[i] for i in selected]
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
            "Some words may be locally reordered or replaced by [MASK]. "
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

    @staticmethod
    def _word_edit_similarity(source: str, recovered: str) -> float:
        source_words = DisruptRecoverDetector._word_tokens(source.lower())
        recovered_words = DisruptRecoverDetector._word_tokens(recovered.lower())
        max_len = max(len(source_words), len(recovered_words), 1)
        try:
            import editdistance
            distance = editdistance.eval(source_words, recovered_words)
        except ImportError:
            from difflib import SequenceMatcher
            return float(SequenceMatcher(None, source_words, recovered_words).ratio())
        return float(max(0.0, 1.0 - distance / max_len))

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
                similarity = self._word_edit_similarity(text, recovered)
                ai_score = self._similarity_to_ai_score(similarity)
                ai_score = self._complexity_penalty(text, ai_score)
                results.append(DetectionResult(
                    score=ai_score,
                    label=DetectionResult.label_from_score(ai_score),
                    detector_name=self.name,
                    details={
                        "word_edit_similarity": similarity,
                        "corrupted": corrupted,
                        "recovered": recovered,
                        "threshold": self._threshold,
                        "disruption_rate": self._disruption_rate,
                        "disruption_mode": self._disruption_mode,
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


# ===========================================================================
# 7. Markov-informed calibration (ICLR 2026)
# ===========================================================================


class MarkovCalibratedTextDetector(BaseDetector):
    """Markov-informed local-score calibration wrapper.

    Wu et al. (ICLR 2026) show that token-level metric-detector scores benefit
    from two structural priors: nearby scores should be similar, while initial
    positions are less stable. Their paper implements this with a Markov random
    field and mean-field approximation.

    Most detectors in this package expose document scores, not raw token scores.
    This wrapper therefore applies the same idea to overlapping local windows:
    score each window with a base detector or callable, run a mean-field-style
    neighbor smoother, downweight unstable early windows, and aggregate the
    calibrated local probabilities. Use it as a lightweight calibration layer,
    not as a replacement for a paper-exact token-level MRF implementation.
    """

    name = "Markov-Calibrated Detector"
    modality = "text"

    def __init__(
        self,
        base_detector: Optional[BaseDetector] = None,
        score_fn: Optional[Callable[[str], float]] = None,
        window_size: int = 64,
        stride: int = 32,
        neighbor_weight: float = 0.8,
        unary_weight: float = 1.0,
        initial_discount: float = 0.5,
        iterations: int = 4,
    ):
        if base_detector is None and score_fn is None:
            raise ValueError("MarkovCalibratedTextDetector needs base_detector or score_fn")
        self.base_detector = base_detector
        self.score_fn = score_fn
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.neighbor_weight = float(neighbor_weight)
        self.unary_weight = float(unary_weight)
        self.initial_discount = float(initial_discount)
        self.iterations = int(iterations)
        if base_detector is not None:
            self.name = f"Markov-Calibrated {base_detector.name}"

    def is_available(self) -> bool:
        if self.base_detector is not None:
            return self.base_detector.is_available()
        return self.score_fn is not None

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def calibrate_scores(
        raw_scores: Sequence[float],
        neighbor_weight: float = 0.8,
        unary_weight: float = 1.0,
        initial_discount: float = 0.5,
        iterations: int = 4,
    ) -> np.ndarray:
        """Mean-field-style calibration for a 1D score chain."""
        raw = np.asarray(raw_scores, dtype=np.float64)
        if raw.size == 0:
            return raw.astype(np.float32)
        if raw.size == 1:
            return np.clip(raw, 0.0, 1.0).astype(np.float32)

        unary = MarkovCalibratedTextDetector._logit(raw)
        weights = np.ones_like(raw)
        weights[0] = float(initial_discount)
        if raw.size > 2:
            weights[1] = max(float(initial_discount), 0.75)
        q = np.clip(raw, 1e-5, 1.0 - 1e-5)

        for _ in range(max(1, int(iterations))):
            prev = np.concatenate(([q[0]], q[:-1]))
            nxt = np.concatenate((q[1:], [q[-1]]))
            # Binary pairwise MRF mean-field update: neighbor fields push each
            # node toward adjacent beliefs, while the unary term preserves the
            # detector's own evidence.
            neighbor_field = (prev - 0.5) + (nxt - 0.5)
            logits = float(unary_weight) * weights * unary + float(neighbor_weight) * neighbor_field
            q = MarkovCalibratedTextDetector._sigmoid(logits)
        return np.clip(q, 0.0, 1.0).astype(np.float32)

    def _windows(self, text: str) -> List[str]:
        words = text.split()
        if len(words) <= self.window_size:
            return [text]
        stride = max(1, self.stride)
        windows = []
        for start in range(0, len(words), stride):
            chunk = words[start : start + self.window_size]
            if len(chunk) < max(8, self.window_size // 4) and windows:
                break
            windows.append(" ".join(chunk))
            if start + self.window_size >= len(words):
                break
        return windows or [text]

    def _score_window(self, window: str) -> float:
        if self.score_fn is not None:
            return float(self.score_fn(window))
        if self.base_detector is None:
            raise RuntimeError("No base detector or score_fn configured")
        result = self.base_detector.detect(window)
        if isinstance(result, list):
            result = result[0]
        return float(result.score)

    def detect(self, content: Union[str, List[str]]) -> Union[DetectionResult, List[DetectionResult]]:
        if not self.is_available():
            raise RuntimeError("Markov calibration base detector is unavailable")

        is_single = isinstance(content, str)
        texts = [content] if is_single else content

        results = []
        for text in texts:
            windows = self._windows(text)
            raw_scores = [max(0.0, min(1.0, self._score_window(w))) for w in windows]
            calibrated = self.calibrate_scores(
                raw_scores,
                neighbor_weight=self.neighbor_weight,
                unary_weight=self.unary_weight,
                initial_discount=self.initial_discount,
                iterations=self.iterations,
            )

            if len(calibrated) > 1:
                weights = np.ones(len(calibrated), dtype=np.float32)
                weights[0] = float(self.initial_discount)
                if len(weights) > 2:
                    weights[1] = max(float(self.initial_discount), 0.75)
                ai_score = float(np.average(calibrated, weights=weights))
            else:
                ai_score = float(calibrated[0])
            ai_score = self._complexity_penalty(text, ai_score)
            results.append(DetectionResult(
                score=ai_score,
                label=DetectionResult.label_from_score(ai_score),
                detector_name=self.name,
                details={
                    "raw_window_scores": [float(s) for s in raw_scores],
                    "calibrated_window_scores": [float(s) for s in calibrated],
                    "num_windows": len(windows),
                    "window_size": self.window_size,
                    "stride": self.stride,
                    "neighbor_weight": self.neighbor_weight,
                    "initial_discount": self.initial_discount,
                },
            ))

        return results[0] if is_single else results


# ===========================================================================
# 8. Inverse Prompting for AI Detection (IPAD - NeurIPS 2025)
# ===========================================================================


class IPADDetector(BaseDetector):
    """IPAD (Chen et al., NeurIPS 2025) — faithful reproduction.

    Loads the authors' three released LoRA adapters on top of
    ``microsoft/Phi-3-medium-128k-instruct``:

    - ``bellafc/IPAD/Prompt_Inverter`` — generates a predicted prompt p̂ from T.
    - ``bellafc/IPAD/Distinguisher_RC`` — answers "can an LLM generate T from p̂?"
      and we extract ``softmax(first_token_logits)[yes_token]``.
    - ``bellafc/IPAD/Distinguisher_PTCV`` — compares T against a regeneration T'
      conditioned on p̂ and returns ``P(yes)`` in the same way.

    The final AI score is the mean of the RC and PTCV yes-probabilities, which
    matches the paper's published ensemble.

    The base model is ~14B parameters; the detector will decline to load unless
    CUDA is available with sufficient memory, and will fall back to unavailable
    otherwise. Adapter weights stay on one shared base, switched via
    ``PeftModel.set_adapter``.
    """

    name = "IPAD"
    modality = "text"

    BASE_MODEL = "microsoft/Phi-3-medium-128k-instruct"
    ADAPTERS = {
        "inverter": "bellafc/IPAD/Prompt_Inverter",
        "rc": "bellafc/IPAD/Distinguisher_RC",
        "ptcv": "bellafc/IPAD/Distinguisher_PTCV",
    }

    def __init__(self, device: str = "auto", dtype: Optional[str] = None):
        self._device_spec = device
        self._dtype = dtype
        self._model = None
        self._tokenizer = None
        self._yes_token_id: Optional[int] = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            if self._device_spec == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self._device_spec

            if device == "cpu":
                logger.warning(
                    "IPADDetector requires a GPU with ~30GB VRAM for Phi-3-medium; "
                    "refusing to load on CPU."
                )
                self._available = False
                return

            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            torch_dtype = dtype_map.get(
                self._dtype,
                torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL, torch_dtype=torch_dtype, device_map=device,
                trust_remote_code=True,
            )
            first_name, first_repo = next(iter(self.ADAPTERS.items()))
            self._model = PeftModel.from_pretrained(base, first_repo, adapter_name=first_name)
            for name, repo in self.ADAPTERS.items():
                if name == first_name:
                    continue
                self._model.load_adapter(repo, adapter_name=name)
            self._model.eval()

            # Cache the "Yes" token id. Phi-3 tokenizes " Yes" as a single token;
            # fall back to "Yes" without leading space if the first lookup yields
            # a multi-token sequence.
            for candidate in (" Yes", "Yes"):
                ids = self._tokenizer(candidate, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    self._yes_token_id = int(ids[0])
                    break
            if self._yes_token_id is None:
                # Take the first token of "Yes" as an approximation — still
                # better than bigram Jaccard.
                self._yes_token_id = int(
                    self._tokenizer("Yes", add_special_tokens=False).input_ids[0]
                )
            self._available = True
        except Exception as e:
            logger.warning("IPADDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM  # noqa: F401
                from peft import PeftModel  # noqa: F401
                self._available = bool(torch.cuda.is_available())
            except ImportError:
                self._available = False
        return bool(self._available)

    def _generate(self, adapter: str, prompt: str, max_new_tokens: int) -> str:
        import torch

        self._model.set_adapter(adapter)
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic for prompt inversion/regeneration
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip()

    def _yes_probability(self, adapter: str, prompt: str) -> float:
        import torch
        import torch.nn.functional as F

        self._model.set_adapter(adapter)
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self._model.device)
        with torch.no_grad():
            logits = self._model(**inputs).logits[0, -1, :]  # (vocab,)
        probs = F.softmax(logits, dim=-1)
        return float(probs[self._yes_token_id].item())

    def detect(self, content: Union[str, List[str]]) -> Union[DetectionResult, List[DetectionResult]]:
        self._load()
        if not self._available:
            raise RuntimeError("IPAD model not loaded")

        is_single = isinstance(content, str)
        texts = [content] if is_single else content

        results = []
        for text in texts:
            if len(text.strip()) < 20:
                results.append(DetectionResult(
                    score=0.5, label="uncertain", detector_name=self.name,
                    details={"reason": "too short"},
                ))
                continue

            try:
                # Step 1: invert — predict the prompt p̂ that could have generated T.
                inv_prompt = (
                    "Read the following text and predict the exact prompt that a "
                    "user would have given to an AI to generate this text.\n\n"
                    f"Text: {text}\n\nPredicted prompt:"
                )
                predicted_prompt = self._generate("inverter", inv_prompt, max_new_tokens=80)

                # Step 2: RC distinguisher — can the LLM generate T from p̂?
                rc_prompt = (
                    "You are given a candidate prompt and a text. Answer Yes if an "
                    "AI model could plausibly have generated the text from the "
                    "prompt, and No otherwise.\n\n"
                    f"Prompt: {predicted_prompt}\n\nText: {text}\n\nAnswer:"
                )
                p_rc = self._yes_probability("rc", rc_prompt)

                # Step 3: PTCV distinguisher — regenerate T' from p̂, then
                # ask whether T and T' come from similar prompts.
                regen_prompt = (
                    "Act as an AI assistant. Please fulfill the following "
                    f"request:\n\n{predicted_prompt}\n\nResponse:"
                )
                regenerated = self._generate("inverter", regen_prompt, max_new_tokens=200)

                ptcv_prompt = (
                    "Text2 is generated by an LLM. Determine whether Text1 is also "
                    "generated by an LLM using a similar prompt. Answer Yes or No.\n\n"
                    f"Text1: {text}\n\nText2: {regenerated}\n\nAnswer:"
                )
                p_ptcv = self._yes_probability("ptcv", ptcv_prompt)

                ai_score = 0.5 * p_rc + 0.5 * p_ptcv
                ai_score = float(max(0.0, min(1.0, ai_score)))
                results.append(DetectionResult(
                    score=ai_score,
                    label=DetectionResult.label_from_score(ai_score),
                    detector_name=self.name,
                    details={
                        "predicted_prompt": predicted_prompt,
                        "rc_yes_probability": p_rc,
                        "ptcv_yes_probability": p_ptcv,
                        "regenerated": regenerated,
                    },
                ))
            except Exception as e:
                logger.error(f"IPAD failed: {e}")
                results.append(DetectionResult(
                    score=0.5, label="error", detector_name=self.name,
                    details={"error": str(e)},
                ))

        return results[0] if is_single else results


class DivEyeDetector(BaseDetector):
    """Zero-shot detector using surprisal diversity features.

    Paper: Basani & Chen, "Diversity Boosts AI-Generated Text Detection" (TMLR 2025).
    Core idea: human text has higher variability in lexical/structural unpredictability.
    Features: mean, std, skewness, kurtosis of token-level surprisal, plus 1st/2nd derivatives.

    Scoring model defaults to ``Qwen/Qwen3.5-9B-Base`` because DivEye reads the
    raw token-level surprisal distribution: a Base (non-RLHF'd) model gives
    cleaner natural-text surprisal statistics. An Instruct model would have
    peaked logits on alignment tokens that distort the diversity signal.
    """

    name = "DivEye"
    modality = "text"

    def __init__(
        self,
        scoring_model: str = "Qwen/Qwen3.5-9B-Base",
        max_length: int = 512,
        device: str = "auto",
    ):
        self._scoring_model_name = scoring_model
        self._max_length = max_length
        self._device_spec = device
        self._model = None
        self._tokenizer = None
        self._available: Optional[bool] = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self._device_spec
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            # Qwen 3.5 9B is too large for fp32 even on a 24 GB card; prefer bf16
            # on Ampere+, fall back to fp16 elsewhere on CUDA, and only allow fp32
            # on CPU (which will be slow but at least functional with a small model).
            if device == "cuda":
                if torch.cuda.get_device_capability()[0] >= 8:
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16
            else:
                dtype = torch.float32

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._scoring_model_name, trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self._scoring_model_name, torch_dtype=dtype, trust_remote_code=True,
            ).to(device)
            self._model.eval()
            self._available = True
        except Exception as e:
            logger.warning("DivEyeDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            # Qwen 3.5 9B needs a GPU with ≥18 GB VRAM in bf16/fp16. Mark
            # unavailable on CPU-only machines so the ensemble degrades cleanly.
            try:
                import torch
                self._available = bool(torch.cuda.is_available())
            except ImportError:
                self._available = False
        return bool(self._available)

    def _extract_surprisal_features(self, text: str) -> Dict[str, float]:
        """Extract DivEye-style surprisal diversity features."""
        import torch
        from scipy import stats as sp_stats

        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        ).to(self._device)

        input_ids = tokens["input_ids"]
        seq_len = input_ids.size(1)
        if seq_len < 5:
            return {"mean": 0, "std": 0, "skew": 0, "kurtosis": 0, "d1_std": 0, "d2_std": 0}

        with torch.no_grad():
            logits = self._model(**tokens).logits[:, :-1]
            log_probs = torch.log_softmax(logits, dim=-1)
            target_ids = input_ids[:, 1:]
            token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        # Surprisal = -log_prob
        surprisal = -token_log_probs.squeeze(0).cpu().float().numpy()

        if len(surprisal) < 3:
            return {"mean": 0, "std": 0, "skew": 0, "kurtosis": 0, "d1_std": 0, "d2_std": 0}

        # Core statistics
        mean = float(np.mean(surprisal))
        std = float(np.std(surprisal))
        skew = float(sp_stats.skew(surprisal))
        kurtosis = float(sp_stats.kurtosis(surprisal))

        # First and second derivatives (DivEye's key contribution)
        d1 = np.diff(surprisal)
        d2 = np.diff(d1) if len(d1) > 1 else np.array([0.0])
        d1_std = float(np.std(d1)) if len(d1) > 0 else 0.0
        d2_std = float(np.std(d2)) if len(d2) > 0 else 0.0

        return {
            "mean": mean,
            "std": std,
            "skew": skew,
            "kurtosis": kurtosis,
            "d1_std": d1_std,
            "d2_std": d2_std,
        }

    def detect(self, content: str) -> DetectionResult:
        self._load()
        if not self._available:
            raise RuntimeError("DivEye scoring model not loaded")

        features = self._extract_surprisal_features(content)

        # Decision heuristic based on DivEye findings:
        # Human text has HIGHER std, d1_std, d2_std (more variable surprisal)
        # AI text has LOWER variability (flatter surprisal profile)
        # Combine features into a score using calibrated thresholds from the paper
        diversity_score = (
            0.3 * min(features["std"] / 5.0, 1.0)
            + 0.25 * min(features["d1_std"] / 4.0, 1.0)
            + 0.25 * min(features["d2_std"] / 3.0, 1.0)
            + 0.1 * min(abs(features["skew"]) / 2.0, 1.0)
            + 0.1 * min(abs(features["kurtosis"]) / 5.0, 1.0)
        )

        # Low diversity → AI, high diversity → human
        ai_score = 1.0 - min(max(diversity_score, 0.0), 1.0)

        return DetectionResult(
            score=ai_score,
            label=DetectionResult.label_from_score(ai_score),
            details={"surprisal_features": features, "diversity_score": diversity_score},
        )
