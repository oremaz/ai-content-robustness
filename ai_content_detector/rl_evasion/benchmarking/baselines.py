"""Baseline evasion methods for benchmarking against RL approaches.

Each baseline implements a common interface so the benchmark runner can
treat all methods uniformly.

Baselines:
    1. Vanilla — raw LLM output, no evasion (lower bound)
    2. Paraphrasing — rewrite with a seq2seq paraphraser (Pegasus/T5)
    3. Synonym substitution — WordNet-based content word replacement
    4. Prompt engineering — instruct the LLM to write "like a human"
    5. Sampling perturbation — high temperature + nucleus sampling
    6. OpenRouter API — any model via OpenRouter (requires API key)
    7. Adversarial Paraphrasing (Chen et al., NeurIPS 2025) — iterative
       detector-guided paraphrasing using an off-the-shelf instruction LLM.
"""

from __future__ import annotations

import logging
import os
import random
import re
from abc import ABC
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class EvasionCapability(str, Enum):
    REWRITE = "rewrite_existing_text"
    GENERATE = "generate_from_prompt"


class BaseEvasionMethod(ABC):
    """Common interface for all evasion methods."""

    name: str = "base"
    capabilities: frozenset[EvasionCapability] = frozenset()
    optimized_detector_names: frozenset[str] = frozenset()

    def provenance(self) -> dict:
        return {
            "method_class": type(self).__name__,
            "capabilities": sorted(capability.value for capability in self.capabilities),
            "optimized_detector_names": sorted(self.optimized_detector_names),
        }

    def evade(self, text: str) -> str:
        """Transform AI-generated text to evade detection.

        Args:
            text: AI-generated input text.

        Returns:
            Transformed text intended to evade detectors.
        """
        raise NotImplementedError(f"{self.name} does not support rewriting existing text")

    def evade_batch(self, texts: List[str]) -> List[str]:
        return [self.evade(t) for t in texts]

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt (for generation-based methods).

        Default: not implemented — subclasses that generate from scratch
        should override this.
        """
        raise NotImplementedError(f"{self.name} does not support generation from prompt")


# ---------------------------------------------------------------------------
# 1. Vanilla (no evasion)
# ---------------------------------------------------------------------------


class VanillaBaseline(BaseEvasionMethod):
    """No evasion — returns text as-is. The detection lower bound."""

    name = "vanilla"
    capabilities = frozenset({EvasionCapability.REWRITE})

    def evade(self, text: str) -> str:
        return text


# ---------------------------------------------------------------------------
# 2. Paraphrasing
# ---------------------------------------------------------------------------


class ParaphrasingBaseline(BaseEvasionMethod):
    """Rewrite text using a seq2seq paraphraser.

    Uses Pegasus-paraphrase (or T5-base) to paraphrase each sentence.
    This is the most common real-world evasion strategy.
    """

    name = "paraphrase"
    capabilities = frozenset({EvasionCapability.REWRITE})

    def __init__(
        self,
        model_name: str = "tuner007/pegasus_paraphrase",
        device: str = "auto",
        max_length: int = 128,
        num_beams: int = 5,
    ):
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._num_beams = num_beams
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name).to(device)
        self._model.eval()
        logger.info("Paraphraser loaded: %s on %s", self._model_name, device)

    def _paraphrase_chunk(self, text: str) -> str:
        import torch

        tokens = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self._max_length
        ).to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **tokens,
                max_length=self._max_length,
                num_beams=self._num_beams,
                num_return_sequences=1,
                early_stopping=True,
            )
        return self._tokenizer.decode(out[0], skip_special_tokens=True)

    def evade(self, text: str) -> str:
        self._load()
        # Split into sentences, paraphrase each, rejoin
        sentences = re.split(r"(?<=[.!?])\s+", text)
        paraphrased = []
        for sent in sentences:
            if len(sent.split()) < 3:
                paraphrased.append(sent)
            else:
                paraphrased.append(self._paraphrase_chunk(sent))
        return " ".join(paraphrased)


# ---------------------------------------------------------------------------
# 3. Synonym substitution
# ---------------------------------------------------------------------------


class SynonymSubstitutionBaseline(BaseEvasionMethod):
    """Replace content words with WordNet synonyms.

    A simple lexical attack that changes surface tokens without
    altering meaning significantly.
    """

    name = "synonym_sub"
    capabilities = frozenset({EvasionCapability.REWRITE})

    def __init__(self, replacement_rate: float = 0.3, seed: int = 42):
        self._rate = replacement_rate
        self._rng = random.Random(seed)
        self._wordnet = None

    def _load_wordnet(self):
        if self._wordnet is not None:
            return
        try:
            from nltk.corpus import wordnet
            # Ensure data is downloaded
            import nltk
            nltk.download("wordnet", quiet=True)
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            self._wordnet = wordnet
        except ImportError:
            raise RuntimeError("NLTK and WordNet are required for synonym substitution")

    def _get_synonym(self, word: str) -> Optional[str]:
        if self._wordnet is None:
            return None
        synsets = self._wordnet.synsets(word)
        if not synsets:
            return None
        synonyms = set()
        for syn in synsets[:3]:
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    synonyms.add(name)
        if not synonyms:
            return None
        return self._rng.choice(list(synonyms))

    def evade(self, text: str) -> str:
        self._load_wordnet()
        if self._wordnet is None:
            raise RuntimeError("WordNet is unavailable")

        words = text.split()
        result = []
        for word in words:
            # Skip short words, punctuation, and apply rate
            clean = re.sub(r"[^\w]", "", word)
            if len(clean) > 3 and self._rng.random() < self._rate:
                syn = self._get_synonym(clean)
                if syn:
                    # Preserve punctuation around the word
                    prefix = word[:len(word) - len(word.lstrip("\"'("))]
                    suffix = word[len(word.rstrip("\"'.,;:!?)")):]
                    result.append(prefix + syn + suffix)
                    continue
            result.append(word)
        return " ".join(result)


# ---------------------------------------------------------------------------
# 4. Prompt engineering
# ---------------------------------------------------------------------------


class PromptEngineeringBaseline(BaseEvasionMethod):
    """Use evasion-oriented system prompts to generate human-sounding text.

    Instructs the model to mimic human writing patterns: varied sentence
    length, informal tone, occasional grammatical imperfections.
    """

    name = "prompt_eng"
    capabilities = frozenset({EvasionCapability.GENERATE})

    EVASION_SYSTEM_PROMPT = (
        "You are a casual human writer. Write naturally with varied sentence lengths. "
        "Use contractions, occasional colloquialisms, and a conversational tone. "
        "Avoid overly formal or structured language. Do not use phrases like "
        "'in conclusion', 'furthermore', or 'it is important to note'. "
        "Occasionally start sentences with 'And' or 'But'. Vary your vocabulary "
        "and avoid repeating the same sentence structure."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B-Base",
        device: str = "auto",
        max_new_tokens: int = 256,
        seed: int = 42,
    ):
        self._model_name = model_name
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._seed = int(seed)
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
        )
        self._model.eval()

    def generate(self, prompt: str) -> str:
        self._load()
        import torch

        full_prompt = f"{self.EVASION_SYSTEM_PROMPT}\n\n{prompt}"
        inputs = self._tokenizer(
            full_prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)

        with torch.no_grad():
            generator = torch.Generator(device=inputs["input_ids"].device).manual_seed(self._seed)
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                generator=generator,
            )
        return self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )


# ---------------------------------------------------------------------------
# 5. Sampling perturbation
# ---------------------------------------------------------------------------


class SamplingPerturbationBaseline(BaseEvasionMethod):
    """Generate text with high temperature + nucleus sampling.

    Higher temperature and broader nucleus sampling produce more
    varied (less predictable) text, closer to human entropy patterns.
    """

    name = "sampling_perturb"
    capabilities = frozenset({EvasionCapability.GENERATE})

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B-Base",
        device: str = "auto",
        temperature: float = 1.2,
        top_p: float = 0.98,
        top_k: int = 100,
        max_new_tokens: int = 256,
        seed: int = 42,
    ):
        self._model_name = model_name
        self._device = device
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._max_new_tokens = max_new_tokens
        self._seed = int(seed)
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
        )
        self._model.eval()

    def generate(self, prompt: str) -> str:
        self._load()
        import torch

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)

        with torch.no_grad():
            generator = torch.Generator(device=inputs["input_ids"].device).manual_seed(self._seed)
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=True,
                temperature=self._temperature,
                top_p=self._top_p,
                top_k=self._top_k,
                generator=generator,
            )
        return self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )


# ---------------------------------------------------------------------------
# 6. OpenRouter API
# ---------------------------------------------------------------------------


class OpenRouterBaseline(BaseEvasionMethod):
    """Generate text via any OpenRouter-hosted LLM.

    Requires the OPENROUTER_API_KEY environment variable.
    Model IDs follow OpenRouter naming, e.g. "google/gemma-3-27b-it".
    See https://openrouter.ai/models for the full catalogue.
    """

    name = "openrouter"
    capabilities = frozenset({EvasionCapability.REWRITE, EvasionCapability.GENERATE})

    def __init__(
        self,
        model_id: str = "google/gemma-3-27b-it",
        max_tokens: int = 256,
        temperature: float = 0.8,
        api_key: Optional[str] = None,
    ):
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.name = f"openrouter({model_id.split('/')[-1]})"

    def _call_api(self, prompt: str, system: str = "") -> str:
        import json
        import urllib.request
        import urllib.error

        if not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. "
                "Export it or pass api_key= to OpenRouterBaseline."
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = json.dumps({
            "model": self._model_id,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }).encode()

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, IndexError) as e:
            raise RuntimeError("OpenRouter API call failed") from e

    def evade(self, text: str) -> str:
        return self._call_api(
            f"Paraphrase the following text naturally:\n\n{text}",
            system="You are a helpful assistant. Rewrite the text in your own words.",
        )

    def generate(self, prompt: str) -> str:
        return self._call_api(prompt)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. Adversarial Paraphrasing (Cheng et al., NeurIPS 2025)
# ---------------------------------------------------------------------------


class AdversarialParaphrasingBaseline(BaseEvasionMethod):
    """Detector-guided token decoding from Cheng et al. (NeurIPS 2025)."""

    name = "adversarial_paraphrasing"
    capabilities = frozenset({EvasionCapability.REWRITE})

    DEFAULT_INSTRUCTION = (
        "Paraphrase the following text. Preserve all factual content and the "
        "overall meaning. Vary sentence structure and word choice. Write in a "
        "natural human voice — avoid the typical patterns of AI-generated "
        "writing (overly balanced sentences, formulaic transitions, hedged "
        "claims, list-of-three structures, mechanical conclusions). Output "
        "only the paraphrase, with no preamble.\n\n"
        "Text:\n{text}\n\nParaphrase:"
    )

    def __init__(
        self,
        guidance_detector: Optional[callable] = None,
        paraphraser_id: str = "Qwen/Qwen3.5-9B",
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.9,
        top_p: float = 0.99,
        top_k: int = 50,
        instruction: Optional[str] = None,
    ):
        if guidance_detector is None:
            raise ValueError(
                "AdversarialParaphrasingBaseline requires a `guidance_detector` "
                "callable (text -> AI probability in [0,1]). Pass e.g. a "
                "BinocularsDetector wrapped as `lambda t: det.detect(t).score`."
            )
        self.guidance_detector = guidance_detector
        self._paraphraser_id = paraphraser_id
        self._device = device
        self._max_new_tokens = int(max_new_tokens)
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._top_k = int(top_k)
        if not 0.0 < self._top_p <= 1.0 or self._top_k < 1:
            raise ValueError("top_p must be in (0,1] and top_k must be positive")
        self._instruction = instruction or self.DEFAULT_INSTRUCTION
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(self._paraphraser_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = (
            torch.bfloat16
            if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16 if device == "cuda" else torch.float32
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._paraphraser_id, torch_dtype=dtype, device_map=device,
        )
        self._model.eval()

    @staticmethod
    def _candidate_token_ids(logits, top_p: float, top_k: int):
        import torch

        probabilities = torch.softmax(logits, dim=-1)
        sorted_probabilities, sorted_ids = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        keep = cumulative <= top_p
        keep[0] = True
        crossing = int(keep.sum().item())
        if crossing < len(keep):
            keep[crossing] = True
        candidate_ids = sorted_ids[keep]
        return candidate_ids[:top_k]

    def _prepare_prompt(self, text: str):
        prompt = self._instruction.format(text=text)
        try:
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return prompt

    def evade(self, text: str) -> str:
        import torch

        if not text or not text.strip():
            return text
        self._load()

        chat = self._prepare_prompt(text)
        inputs = self._tokenizer(
            chat, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self._device)
        prompt_len = inputs["input_ids"].shape[1]
        sequence = inputs["input_ids"]
        attention = inputs.get("attention_mask")
        for _ in range(self._max_new_tokens):
            with torch.no_grad():
                outputs = self._model(input_ids=sequence, attention_mask=attention)
            logits = outputs.logits[0, -1] / self._temperature
            candidate_ids = self._candidate_token_ids(logits, self._top_p, self._top_k)
            scored = []
            for token_id in candidate_ids.tolist():
                candidate_output = torch.cat([
                    sequence[0, prompt_len:],
                    torch.tensor([token_id], device=sequence.device),
                ])
                candidate_text = self._tokenizer.decode(
                    candidate_output, skip_special_tokens=True,
                )
                score = float(self.guidance_detector(candidate_text))
                if not 0.0 <= score <= 1.0:
                    raise ValueError(f"Guidance detector returned invalid score {score}")
                scored.append((score, token_id))
            if not scored:
                break
            _, selected_id = min(scored, key=lambda pair: pair[0])
            selected = torch.tensor([[selected_id]], device=sequence.device)
            sequence = torch.cat([sequence, selected], dim=1)
            attention = torch.ones_like(sequence)
            if selected_id == self._tokenizer.eos_token_id:
                break
        return self._tokenizer.decode(
            sequence[0, prompt_len:], skip_special_tokens=True,
        ).strip()


def get_all_baselines(device: str = "auto", model_name: str = "Qwen/Qwen3.5-9B-Base") -> List[BaseEvasionMethod]:
    """Return all baseline evasion methods."""
    methods = [
        VanillaBaseline(),
        ParaphrasingBaseline(device=device),
        SynonymSubstitutionBaseline(),
        PromptEngineeringBaseline(model_name=model_name, device=device),
        SamplingPerturbationBaseline(model_name=model_name, device=device),
    ]
    # Include OpenRouter if API key is available
    if os.environ.get("OPENROUTER_API_KEY"):
        methods.append(OpenRouterBaseline())
    return methods


def get_lightweight_baselines() -> List[BaseEvasionMethod]:
    """Return baselines that don't require GPU or large models."""
    methods = [
        VanillaBaseline(),
        SynonymSubstitutionBaseline(),
    ]
    if os.environ.get("OPENROUTER_API_KEY"):
        methods.append(OpenRouterBaseline())
    return methods
