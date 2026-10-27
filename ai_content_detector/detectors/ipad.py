"""Inverse prompting with the released IPAD adapters."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Union

import numpy as np

from .ensemble import BaseDetector, DetectionResult
import os
from .text_detectors_utils import _check_hf_config

logger = logging.getLogger("ai_content_detector.detectors.text_detectors")


class IPADDetector(BaseDetector):
    """IPAD (Chen et al., NeurIPS 2025): faithful reproduction.

    Loads the authors' three released LoRA adapters on top of
    ``microsoft/Phi-3-medium-128k-instruct``:

    - ``bellafc/IPAD/Prompt_Inverter``: generates a predicted prompt p̂ from T.
    - ``bellafc/IPAD/Distinguisher_PTCV`` checks prompt/text consistency.
    - ``bellafc/IPAD/Distinguisher_RC`` compares the input and regeneration.

    Yes/no sequence likelihoods are renormalized as a binary distribution. The
    paper fusion weight (0.45 for PTCV) and threshold (0.54) are defaults.

    The base model is ~14B parameters; the detector will decline to load unless
    CUDA is available with sufficient memory, and will fall back to unavailable
    otherwise. Adapter weights stay on one shared base, switched via
    ``PeftModel.set_adapter``.
    """

    __module__ = "ai_content_detector.detectors.text_detectors"

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
                # Step 1: invert: predict the prompt p̂ that could have generated T.
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

