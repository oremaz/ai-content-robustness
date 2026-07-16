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


def _check_hf_config(instance, model_name: str, trust_remote_code: bool = False) -> bool:
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        return True
    except Exception as error:
        instance._availability_error = str(error)
        return False

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
        calibration_scale: float = 0.05,
        calibrator: Optional[Callable[[float], float]] = None,
    ):
        self._observer_name = observer_name
        self._performer_name = performer_name
        self._max_length = max_length
        self._device_spec = device
        self._threshold = threshold
        self._calibration_scale = float(calibration_scale)
        self._calibrator = calibrator
        if self._calibration_scale <= 0:
            raise ValueError("calibration_scale must be positive")
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
                self._available = _check_hf_config(self, self._observer_name) and _check_hf_config(
                    self, self._performer_name,
                )
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

        results = []
        for raw_score in raw_scores:
            paper_decision = raw_score < self._threshold
            if self._calibrator is not None:
                ai_score = float(self._calibrator(raw_score))
                score_semantics = "calibrated_probability"
            else:
                ai_score = float(
                    1.0 / (1.0 + np.exp((raw_score - self._threshold) / self._calibration_scale))
                )
                score_semantics = "uncalibrated_monotonic_transform"
            if not 0.0 <= ai_score <= 1.0:
                raise ValueError(f"Binoculars calibrator returned invalid probability: {ai_score}")

            results.append(DetectionResult(
                score=ai_score,
                label="ai" if paper_decision else "human",
                detector_name=self.name,
                details={
                    "raw_binoculars_score": raw_score,
                    "threshold_used": self._threshold,
                    "below_threshold": paper_decision,
                    "paper_decision": "ai" if paper_decision else "human",
                    "score_semantics": score_semantics,
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
        reference_model: Optional[str] = None,
        max_length: int = 512,
        device: str = "auto",
        threshold: float = 0.0,
    ):
        self._scoring_model_name = scoring_model
        self._reference_model_name = reference_model or scoring_model
        self._max_length = max_length
        self._device_spec = device
        self._threshold = float(threshold)
        self._model = None
        self._reference_model = None
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
            if self._reference_model_name == self._scoring_model_name:
                self._reference_model = self._model
            else:
                reference_tokenizer = AutoTokenizer.from_pretrained(self._reference_model_name)
                if reference_tokenizer.get_vocab() != self._tokenizer.get_vocab():
                    raise ValueError(
                        "Fast-DetectGPT scoring and reference models must use identical token vocabularies"
                    )
                self._reference_model = AutoModelForCausalLM.from_pretrained(
                    self._reference_model_name, torch_dtype=dtype, device_map=device,
                )
                self._reference_model.eval()
            self._model.eval()
            self._available = True
        except Exception as e:
            logger.warning("FastDetectGPTDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import torch
                self._available = bool(
                    torch.cuda.is_available()
                    and _check_hf_config(self, self._scoring_model_name)
                    and _check_hf_config(self, self._reference_model_name)
                )
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
            reference_outputs = self._reference_model(**tokens)
            scoring_logits = outputs.logits[:, :-1]
            reference_logits = reference_outputs.logits[:, :-1]
            target_ids = input_ids[:, 1:]  # (batch, seq_len-1)
            criterion = self._criterion_from_logits(
                scoring_logits, reference_logits, target_ids, attention_mask,
            )
            scores = criterion.cpu().numpy().tolist()

        return scores[0] if is_single else scores

    @staticmethod
    def _criterion_from_logits(scoring_logits, reference_logits, target_ids, attention_mask):
        """Compute the standardized Fast-DetectGPT conditional curvature."""
        import torch

        scoring_log_probs = torch.log_softmax(scoring_logits, dim=-1)
        reference_probs = torch.softmax(reference_logits, dim=-1)
        observed = scoring_log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        expected = (reference_probs * scoring_log_probs).sum(dim=-1)
        second_moment = (reference_probs * scoring_log_probs.square()).sum(dim=-1)
        variance = (second_moment - expected.square()).clamp(min=0.0)
        mask = attention_mask.to(scoring_log_probs.dtype)
        numerator = ((observed - expected) * mask).sum(dim=-1)
        denominator = torch.sqrt((variance * mask).sum(dim=-1).clamp(min=1e-12))
        return numerator / denominator

    def detect(self, content: Union[str, List[str]]) -> Union[DetectionResult, List[DetectionResult]]:
        self._load()
        if not self._available:
            raise RuntimeError("Fast-DetectGPT model not loaded")

        is_single = isinstance(content, str)
        texts = [content] if is_single else content
        # texts is always a list, so _compute_curvature returns a list.
        raw_curvatures = self._compute_curvature(texts)

        results = []
        for text, raw_curvature in zip(texts, raw_curvatures):
            ai_score = float(1.0 / (1.0 + np.exp(-(raw_curvature - self._threshold))))
            label = "ai" if raw_curvature >= self._threshold else "human"

            results.append(DetectionResult(
                score=ai_score,
                label=label,
                detector_name=self.name,
                details={
                    "raw_curvature": raw_curvature,
                    "threshold_used": self._threshold,
                    "score_semantics": "uncalibrated_logistic_transform",
                    "reference_model": self._reference_model_name,
                    "scoring_model": self._scoring_model_name,
                },
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

    Word-chunk shuffling, constrained recovery, BERTScore, and word-order rank
    correlations follow the paper protocol.
    """

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


# ===========================================================================
# 7. Markov-informed calibration (ICLR 2026)
# ===========================================================================


class WindowSmoothedTextDetector(BaseDetector):
    """Fixed local-window neighbor smoother for detector scores.

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

    name = "Window-Smoothed Detector"
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
            raise ValueError("WindowSmoothedTextDetector needs base_detector or score_fn")
        self.base_detector = base_detector
        self.score_fn = score_fn
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.neighbor_weight = float(neighbor_weight)
        self.unary_weight = float(unary_weight)
        self.initial_discount = float(initial_discount)
        self.iterations = int(iterations)
        if base_detector is not None:
            self.name = f"Window-Smoothed {base_detector.name}"

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

        unary = WindowSmoothedTextDetector._logit(raw)
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
            q = WindowSmoothedTextDetector._sigmoid(logits)
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


# Compatibility alias. This is not the learned token-level MRF from Wu et al.
MarkovCalibratedTextDetector = WindowSmoothedTextDetector


# ===========================================================================
# 8. Inverse Prompting for AI Detection (IPAD - NeurIPS 2025)
# ===========================================================================


class IPADDetector(BaseDetector):
    """IPAD (Chen et al., NeurIPS 2025) — faithful reproduction.

    Loads the authors' three released LoRA adapters on top of
    ``microsoft/Phi-3-medium-128k-instruct``:

    - ``bellafc/IPAD/Prompt_Inverter`` — generates a predicted prompt p̂ from T.
    - ``bellafc/IPAD/Distinguisher_PTCV`` checks prompt/text consistency.
    - ``bellafc/IPAD/Distinguisher_RC`` compares the input and regeneration.

    Yes/no sequence likelihoods are renormalized as a binary distribution. The
    paper fusion weight (0.45 for PTCV) and threshold (0.54) are defaults.

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

    def __init__(
        self,
        device: str = "auto",
        dtype: Optional[str] = None,
        regenerator: Optional[Callable[[str], str]] = None,
        regeneration_model: str = "gpt-3.5-turbo",
        fusion_weight: float = 0.45,
        threshold: float = 0.54,
    ):
        self._device_spec = device
        self._dtype = dtype
        self._model = None
        self._tokenizer = None
        self._regenerator = regenerator
        self._regeneration_model = regeneration_model
        self._fusion_weight = float(fusion_weight)
        self._threshold = float(threshold)
        if not 0.0 <= self._fusion_weight <= 1.0:
            raise ValueError("fusion_weight must be in [0, 1]")
        if not 0.0 <= self._threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
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

            self._available = True
        except Exception as e:
            logger.warning("IPADDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import torch
                from peft import PeftConfig
                if not torch.cuda.is_available():
                    self._availability_error = "CUDA is required for the Phi-3-medium IPAD model"
                    self._available = False
                else:
                    self._available = _check_hf_config(
                        self, self.BASE_MODEL, trust_remote_code=True,
                    )
                    if self._available:
                        for repository in self.ADAPTERS.values():
                            PeftConfig.from_pretrained(repository)
            except Exception as error:
                self._availability_error = str(error)
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

    def _answer_log_probability(self, adapter: str, prompt: str, answer: str) -> float:
        import torch

        self._model.set_adapter(adapter)
        prompt_ids = self._tokenizer(
            prompt, add_special_tokens=True, truncation=True, max_length=4088,
        ).input_ids
        answer_ids = self._tokenizer(
            f" {answer}", add_special_tokens=False,
        ).input_ids
        if not answer_ids:
            raise RuntimeError(f"Tokenizer produced no tokens for answer {answer!r}")
        input_ids = torch.tensor([prompt_ids + answer_ids], device=self._model.device)
        with torch.no_grad():
            logits = self._model(input_ids=input_ids).logits[:, :-1]
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = input_ids[:, 1:]
        selected = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        start = len(prompt_ids) - 1
        return float(selected[:, start:].sum().item())

    def _yes_probability(self, adapter: str, prompt: str) -> float:
        yes_logp = self._answer_log_probability(adapter, prompt, "yes")
        no_logp = self._answer_log_probability(adapter, prompt, "no")
        normalizer = np.logaddexp(yes_logp, no_logp)
        return float(np.exp(yes_logp - normalizer))

    def _regenerate(self, prompt: str, max_new_tokens: int = 200) -> tuple[str, str]:
        if self._regenerator is not None:
            return self._regenerator(prompt), "callable"

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            import json
            import urllib.request

            body = json.dumps({
                "model": self._regeneration_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0,
            }).encode()
            request = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read())
            return payload["choices"][0]["message"]["content"].strip(), self._regeneration_model

        disable_adapter = getattr(self._model, "disable_adapter", None)
        if not callable(disable_adapter):
            raise RuntimeError(
                "IPAD regeneration requires a regenerator callable, OPENAI_API_KEY, "
                "or a PEFT model supporting disable_adapter()"
            )
        import torch
        with disable_adapter():
            inputs = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=4096,
            ).to(self._model.device)
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                )
        text = self._tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip()
        return text, "phi3_base_fallback"

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

                # Step 2: prompt-text consistency verifier on (P, T).
                ptcv_prompt = (
                    "You are given a candidate prompt and a text. Answer Yes if an "
                    "AI model could plausibly have generated the text from the "
                    "prompt, and No otherwise.\n\n"
                    f"Prompt: {predicted_prompt}\n\nText: {text}\n\nAnswer:"
                )
                p_ptcv = self._yes_probability("ptcv", ptcv_prompt)

                # Step 3: regenerate from P and compare (T', T) with RC.
                regenerated, regeneration_backend = self._regenerate(predicted_prompt)

                rc_prompt = (
                    "Text2 is generated by an LLM. Determine whether Text1 is also "
                    "generated by an LLM using a similar prompt. Answer Yes or No.\n\n"
                    f"Text1: {text}\n\nText2: {regenerated}\n\nAnswer:"
                )
                p_rc = self._yes_probability("rc", rc_prompt)

                ai_score = self._fusion_weight * p_ptcv + (1.0 - self._fusion_weight) * p_rc
                ai_score = float(max(0.0, min(1.0, ai_score)))
                results.append(DetectionResult(
                    score=ai_score,
                    label="ai" if ai_score > self._threshold else "human",
                    detector_name=self.name,
                    details={
                        "predicted_prompt": predicted_prompt,
                        "rc_yes_probability": p_rc,
                        "ptcv_yes_probability": p_ptcv,
                        "regenerated": regenerated,
                        "regeneration_backend": regeneration_backend,
                        "fusion_weight_ptcv": self._fusion_weight,
                        "decision_threshold": self._threshold,
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
    """DivEye surprisal features with a trained XGBoost classifier.

    Paper: Basani & Chen, "Diversity Boosts AI-Generated Text Detection" (TMLR 2025).
    Core idea: human text has higher variability in lexical/structural unpredictability.
    The nine features follow Equation 6. Detection requires a fitted classifier;
    the features are zero-shot, but the paper's final classifier is supervised.

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
        max_length: int = 1024,
        device: str = "auto",
        classifier_path: Optional[str] = None,
        classifier: Any = None,
        entropy_bins: int = 20,
    ):
        self._scoring_model_name = scoring_model
        self._max_length = max_length
        self._device_spec = device
        self._classifier_path = classifier_path
        self._classifier = classifier
        self._entropy_bins = int(entropy_bins)
        if self._entropy_bins < 2:
            raise ValueError("entropy_bins must be at least 2")
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
            if self._classifier is None and self._classifier_path:
                import joblib
                self._classifier = joblib.load(self._classifier_path)
            self._available = self._classifier is not None
            if self._classifier is None:
                logger.warning("DivEye requires a trained XGBoost classifier")
        except Exception as e:
            logger.warning("DivEyeDetector unavailable: %s", e)
            self._available = False

    def is_available(self) -> bool:
        if self._available is None:
            # Qwen 3.5 9B needs a GPU with ≥18 GB VRAM in bf16/fp16. Mark
            # unavailable on CPU-only machines so the ensemble degrades cleanly.
            try:
                import torch
                compute_available = torch.cuda.is_available() or self._device_spec == "cpu"
                self._available = bool(
                    compute_available and (self._classifier is not None or (
                        self._classifier_path and os.path.exists(self._classifier_path)
                    ))
                )
            except ImportError:
                self._available = False
        return bool(self._available)

    def _extract_surprisal_features(self, text: str) -> Dict[str, float]:
        """Extract DivEye-style surprisal diversity features."""
        import torch

        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        ).to(self._device)

        input_ids = tokens["input_ids"]
        seq_len = input_ids.size(1)
        if seq_len < 5:
            raise ValueError("DivEye requires at least five tokens")

        with torch.no_grad():
            logits = self._model(**tokens).logits[:, :-1]
            log_probs = torch.log_softmax(logits, dim=-1)
            target_ids = input_ids[:, 1:]
            token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        # Surprisal = -log_prob
        surprisal = -token_log_probs.squeeze(0).cpu().float().numpy()

        if len(surprisal) < 3:
            raise ValueError("DivEye requires at least four surprisal values")

        return self._features_from_surprisal(surprisal, self._entropy_bins)

    @staticmethod
    def _features_from_surprisal(surprisal: np.ndarray, entropy_bins: int = 20) -> Dict[str, float]:
        from scipy import stats as sp_stats

        surprisal = np.asarray(surprisal, dtype=np.float64)
        if len(surprisal) < 4:
            raise ValueError("DivEye requires at least four surprisal values")
        mean = float(np.mean(surprisal))
        variance = float(np.var(surprisal, ddof=1))
        if variance == 0.0:
            skew = 0.0
            kurtosis = 0.0
        else:
            skew = float(np.nan_to_num(sp_stats.skew(surprisal), nan=0.0))
            kurtosis = float(np.nan_to_num(sp_stats.kurtosis(surprisal), nan=0.0))

        # First and second derivatives (DivEye's key contribution)
        d1 = np.diff(surprisal)
        d2 = np.diff(d1) if len(d1) > 1 else np.array([0.0])
        d1_mean = float(np.mean(d1))
        d1_variance = float(np.var(d1, ddof=1)) if len(d1) > 1 else 0.0
        d2_variance = float(np.var(d2, ddof=1)) if len(d2) > 1 else 0.0
        counts, _ = np.histogram(d2, bins=entropy_bins)
        probabilities = counts[counts > 0].astype(np.float64)
        probabilities /= probabilities.sum()
        d2_entropy = float(-(probabilities * np.log(probabilities)).sum())
        if len(d2) > 1 and d2_variance > 0:
            centered = d2 - np.mean(d2)
            d2_autocorrelation = float(
                np.mean(centered[:-1] * centered[1:]) / np.mean(np.square(centered))
            )
        else:
            d2_autocorrelation = 0.0

        return {
            "mean": mean,
            "variance": variance,
            "skew": skew,
            "kurtosis": kurtosis,
            "d1_mean": d1_mean,
            "d1_variance": d1_variance,
            "d2_variance": d2_variance,
            "d2_entropy": d2_entropy,
            "d2_autocorrelation": d2_autocorrelation,
        }

    @staticmethod
    def _feature_vector(features: Dict[str, float]) -> np.ndarray:
        names = (
            "mean", "variance", "skew", "kurtosis", "d1_mean",
            "d1_variance", "d2_variance", "d2_entropy", "d2_autocorrelation",
        )
        vector = np.asarray([features[name] for name in names], dtype=np.float32)
        if not np.isfinite(vector).all():
            raise ValueError("DivEye produced non-finite features")
        return vector

    def fit_classifier(self, texts: Sequence[str], labels: Sequence[int], save_path: Optional[str] = None):
        """Fit the paper XGBoost model with labels 0=AI and 1=human."""
        if len(texts) != len(labels) or len(texts) < 4 or set(labels) != {0, 1}:
            raise ValueError("DivEye training needs aligned texts with both labels 0=AI and 1=human")
        if self._model is None:
            existing_classifier = self._classifier
            self._classifier = object()
            self._load()
            self._classifier = existing_classifier
        from xgboost import XGBClassifier

        y = np.asarray(labels, dtype=np.int64)
        positives = max(1, int((y == 1).sum()))
        negatives = int((y == 0).sum())
        classifier = XGBClassifier(
            random_state=42,
            scale_pos_weight=negatives / positives,
            max_depth=12,
            n_estimators=200,
            colsample_bytree=0.8,
            subsample=0.7,
            min_child_weight=5,
            gamma=1.0,
            eval_metric="logloss",
        )
        matrix = np.vstack([
            self._feature_vector(self._extract_surprisal_features(text)) for text in texts
        ])
        classifier.fit(matrix, y)
        self._classifier = classifier
        self._available = True
        if save_path:
            import joblib
            joblib.dump(classifier, save_path)
            self._classifier_path = save_path
        return self

    def detect(self, content: str) -> DetectionResult:
        self._load()
        if not self._available:
            raise RuntimeError("DivEye scoring model not loaded")

        features = self._extract_surprisal_features(content)
        vector = self._feature_vector(features).reshape(1, -1)
        probabilities = self._classifier.predict_proba(vector)[0]
        classes = list(self._classifier.classes_)
        if 0 not in classes:
            raise RuntimeError("DivEye classifier must use the paper labels 0=AI and 1=human")
        ai_score = float(probabilities[classes.index(0)])

        return DetectionResult(
            score=ai_score,
            label=DetectionResult.label_from_score(ai_score),
            detector_name=self.name,
            details={
                "surprisal_features": features,
                "classifier_path": self._classifier_path,
                "score_semantics": "classifier_probability",
            },
        )
