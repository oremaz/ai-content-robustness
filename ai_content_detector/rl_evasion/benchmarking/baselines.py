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
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)


class BaseEvasionMethod(ABC):
    """Common interface for all evasion methods."""

    name: str = "base"

    @abstractmethod
    def evade(self, text: str) -> str:
        """Transform AI-generated text to evade detection.

        Args:
            text: AI-generated input text.

        Returns:
            Transformed text intended to evade detectors.
        """
        ...

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
            logger.warning("NLTK not installed — synonym substitution will be a no-op")
            self._wordnet = None

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
            return text

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
    ):
        self._model_name = model_name
        self._device = device
        self._max_new_tokens = max_new_tokens
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

    def evade(self, text: str) -> str:
        """Not applicable — use generate() instead."""
        return text

    def generate(self, prompt: str) -> str:
        self._load()
        import torch

        full_prompt = f"{self.EVASION_SYSTEM_PROMPT}\n\n{prompt}"
        inputs = self._tokenizer(
            full_prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
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

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B-Base",
        device: str = "auto",
        temperature: float = 1.2,
        top_p: float = 0.98,
        top_k: int = 100,
        max_new_tokens: int = 256,
    ):
        self._model_name = model_name
        self._device = device
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._max_new_tokens = max_new_tokens
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

    def evade(self, text: str) -> str:
        """Not applicable — use generate() instead."""
        return text

    def generate(self, prompt: str) -> str:
        self._load()
        import torch

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=True,
                temperature=self._temperature,
                top_p=self._top_p,
                top_k=self._top_k,
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
            logger.warning("OpenRouter API call failed: %s", e)
            return prompt  # fallback to echo

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
# 7. Adversarial Paraphrasing (Chen et al., NeurIPS 2025)
# ---------------------------------------------------------------------------


class AdversarialParaphrasingBaseline(BaseEvasionMethod):
    """Iterative detector-guided paraphrasing — faithful port of the published
    algorithm (Chen et al., *Adversarial Paraphrasing*, NeurIPS 2025;
    https://github.com/chengez/Adversarial-Paraphrasing).

    Algorithm (per input text T):

        best ← T
        for k in 1..K:
            candidates ← paraphraser.generate(T, n=N)
            scores     ← [guidance_detector(c) for c in candidates]
            best       ← candidates[argmin(scores)]    # lowest AI-prob wins
            T          ← best
            if guidance_detector(T) < tau:
                break
        return best

    The original implementation uses an off-the-shelf instruction-tuned LLM as
    the paraphraser; we default to ``Qwen/Qwen3.5-9B`` (the Instruct variant)
    so the paraphraser shares its base model family with the rest of the
    project's defaults. The caller can swap it via ``paraphraser_id=``. The
    guidance detector is any callable that maps text to an AI probability in
    [0, 1] — passing one of our own detectors makes this a strict baseline
    against which our RL attack should be measured.

    This is *the* strongest training-free 2025 humanizing attack and the right
    yardstick for the paper's headline comparison ("our RL attack vs. the best
    peer-reviewed training-free baseline").
    """

    name = "adversarial_paraphrasing"

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
        max_iterations: int = 5,
        candidates_per_iter: int = 4,
        tau: float = 0.3,
        max_new_tokens: int = 512,
        temperature: float = 0.9,
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
        self.max_iterations = int(max_iterations)
        self.candidates_per_iter = int(candidates_per_iter)
        self.tau = float(tau)
        self._max_new_tokens = int(max_new_tokens)
        self._temperature = float(temperature)
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

    def _paraphrase_candidates(self, text: str) -> List[str]:
        import torch

        prompt = self._instruction.format(text=text)
        # Use chat template if available — both Llama-3 Instruct and most modern
        # instruct models expect it; falling back to raw prompt for older models.
        try:
            chat = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            chat = prompt

        inputs = self._tokenizer(
            chat, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self._device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=True,
                temperature=self._temperature,
                top_p=0.95,
                num_return_sequences=self.candidates_per_iter,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        # out: (N, prompt_len + new_tokens). Drop the prompt prefix per row.
        prompt_len = inputs["input_ids"].shape[1]
        candidates = [
            self._tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            for seq in out
        ]
        # Drop empties — they're useless guidance candidates.
        return [c for c in candidates if c]

    def evade(self, text: str) -> str:
        if not text or not text.strip():
            return text

        self._load()
        best = text
        try:
            best_score = float(self.guidance_detector(best))
        except Exception as e:
            logger.error("AdversarialParaphrasing: guidance_detector failed: %s", e)
            return text

        for it in range(self.max_iterations):
            try:
                candidates = self._paraphrase_candidates(best)
            except Exception as e:
                logger.error("AdversarialParaphrasing: paraphraser failed: %s", e)
                break
            if not candidates:
                break

            scored = []
            for c in candidates:
                try:
                    s = float(self.guidance_detector(c))
                except Exception as e:
                    logger.warning("AdversarialParaphrasing: detector error on candidate: %s", e)
                    continue
                scored.append((s, c))
            if not scored:
                break

            scored.sort(key=lambda sc: sc[0])  # ascending by AI score
            cand_score, cand_text = scored[0]

            # Greedy improvement: only accept candidates that improve on the
            # current best to avoid oscillation.
            if cand_score < best_score:
                best, best_score = cand_text, cand_score
            else:
                # Even when no candidate improves, accept the best one to keep
                # exploring (the paper allows non-monotonic moves to escape
                # local optima).
                best = cand_text
                best_score = cand_score

            if best_score < self.tau:
                logger.info(
                    "AdversarialParaphrasing: converged at iter %d (score=%.3f < tau=%.3f)",
                    it + 1, best_score, self.tau,
                )
                break

        return best


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
