"""Zero-shot text statistics from observer and reference language models."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Union

import numpy as np

from .ensemble import BaseDetector, DetectionResult
from .text_detectors_utils import _check_hf_config

logger = logging.getLogger("ai_content_detector.detectors.text_detectors")


class BinocularsDetector(BaseDetector):
    """Zero-shot detector using cross-perplexity ratio of two reference LMs.

    Paper: Hans et al., "Spotting LLMs With Binoculars" (ICML 2024).
    Core idea: score = perplexity(observer) / cross_perplexity(performer, observer).
    Low score → likely AI-generated.
    """

    __module__ = "ai_content_detector.detectors.text_detectors"

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

        # Numerator: observer perplexity: the observer's per-token cross-entropy
        # against the *actual* tokens (how surprised the observer is by the text).
        ce_observer = F.cross_entropy(
            logits_observer.transpose(1, 2),
            target_ids,
            reduction="none",
        )

        # Denominator: cross-perplexity: the per-token cross-entropy between the
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


class FastDetectGPTDetector(BaseDetector):
    """Zero-shot detector using conditional probability curvature.

    Paper: Bao et al., "Fast-DetectGPT: Efficient Zero-Shot Detection via
    Conditional Probability Curvature" (ICLR 2024).
    """

    __module__ = "ai_content_detector.detectors.text_detectors"

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

