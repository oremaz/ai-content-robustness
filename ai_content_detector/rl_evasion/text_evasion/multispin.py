"""MultiSPIN: Multi-feature distribution matching via self-play.

Extends SPIN (Self-Play Fine-Tuning, Chen et al. 2024) with explicit
multi-feature distribution matching across stylometric, embedding,
and perplexity-curvature feature spaces. The goal is to close the
distributional gap between model output and human text in the feature
spaces that real detectors use.

Novel contribution from todo/rl_evasion_research_directions_v2.md:
    L_MultiSPIN = L_SPIN + λ1*||φ_stylo(y)-φ_stylo(y')||² +
                  λ2*MMD(φ_emb(y),φ_emb(y')) + λ3*L_task

References:
    - Chen et al., SPIN (arXiv:2401.01335)
    - Rafailov et al., DPO (NeurIPS 2023)
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MultiSPINConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------


class StylometricExtractor:
    """Extract stylometric features used by StyloAI-class detectors.

    Features: burstiness, TTR, avg/std sentence length, function word ratio,
    POS bigram entropy, punctuation density, hapax legomena ratio.
    """

    def __init__(self):
        self._nlp = None

    def _load_spacy(self):
        if self._nlp is not None:
            return
        import spacy
        try:
            self._nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            try:
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            except Exception:
                logger.warning(
                    "spaCy model en_core_web_sm is unavailable; falling back to blank English tokenizer. "
                    "POS-entropy features will be zero until the model is installed."
                )
                self._nlp = spacy.blank("en")
                if "sentencizer" not in self._nlp.pipe_names:
                    self._nlp.add_pipe("sentencizer")

    def extract(self, text: str) -> np.ndarray:
        """Extract stylometric feature vector."""
        self._load_spacy()
        from collections import Counter

        doc = self._nlp(text)
        words = [t.text.lower() for t in doc if t.is_alpha]
        n_words = max(len(words), 1)
        n_chars = max(len(text), 1)

        # TTR
        ttr = len(set(words)) / n_words

        # Sentence lengths
        sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
        sent_lens = [len([t for t in s if t.is_alpha]) for s in sents]
        avg_sl = np.mean(sent_lens) if sent_lens else 0
        std_sl = np.std(sent_lens) if len(sent_lens) > 1 else 0
        burstiness = std_sl / avg_sl if avg_sl > 0 else 0

        # Function word ratio
        fw = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "and", "or", "but", "not", "that", "this",
            "it", "he", "she", "they", "we", "you", "i", "my", "his", "her",
        }
        fw_ratio = sum(1 for w in words if w in fw) / n_words

        # POS bigram entropy
        pos_tags = [t.pos_ for t in doc if t.is_alpha]
        if len(pos_tags) > 1:
            bigrams = Counter((pos_tags[i], pos_tags[i + 1]) for i in range(len(pos_tags) - 1))
            total = sum(bigrams.values())
            probs = [c / total for c in bigrams.values()]
            pos_ent = -sum(p * np.log2(p) for p in probs if p > 0)
        else:
            pos_ent = 0

        # Punctuation density
        punct = sum(1 for c in text if c in ".,;:!?-()\"'")
        punct_density = punct / n_chars

        # Hapax ratio
        word_counts = Counter(words)
        hapax_ratio = sum(1 for c in word_counts.values() if c == 1) / n_words

        return np.array([
            burstiness, ttr, avg_sl, std_sl, fw_ratio,
            pos_ent, punct_density, hapax_ratio,
        ], dtype=np.float32)

    def extract_batch(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.extract(t) for t in texts])


class EmbeddingExtractor:
    """Extract sentence embeddings using a frozen encoder (E5, SBERT)."""

    def __init__(self, model_name: str = "intfloat/e5-base-v2", device: str = "auto"):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(device)
        self._model.eval()

    def extract(self, text: str) -> np.ndarray:
        self._load()
        tokens = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self._device)
        with torch.no_grad():
            emb = self._model(**tokens).last_hidden_state[:, 0].cpu().numpy()
        return emb.flatten() / (np.linalg.norm(emb) + 1e-8)

    def extract_batch(self, texts: List[str]) -> np.ndarray:
        self._load()
        tokens = self._tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self._device)
        with torch.no_grad():
            emb = self._model(**tokens).last_hidden_state[:, 0].cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return emb / norms


# ---------------------------------------------------------------------------
# MMD (Maximum Mean Discrepancy) for distribution matching
# ---------------------------------------------------------------------------


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """Compute MMD² between two sample sets using RBF kernel.

    MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    """
    from scipy.spatial.distance import cdist

    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        return 0.0

    XX = cdist(X, X, "sqeuclidean")
    YY = cdist(Y, Y, "sqeuclidean")
    XY = cdist(X, Y, "sqeuclidean")

    K_XX = np.exp(-gamma * XX)
    K_YY = np.exp(-gamma * YY)
    K_XY = np.exp(-gamma * XY)

    # Unbiased estimate
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    mmd2 = K_XX.sum() / (n * (n - 1) + 1e-8) + K_YY.sum() / (m * (m - 1) + 1e-8) - 2 * K_XY.mean()
    return float(max(mmd2, 0.0))


# ---------------------------------------------------------------------------
# MultiSPIN Trainer
# ---------------------------------------------------------------------------


class MultiSPINTrainer:
    """Train a model using MultiSPIN: SPIN + multi-feature distribution matching.

    Training loop:
    1. Generate outputs y' from current model given prompts x.
    2. Compute DPO-style SPIN loss: model learns to prefer human text over its own.
    3. Compute feature-matching losses across stylometric and embedding spaces.
    4. Update model on composite loss.
    5. After each iteration, update persistent feature bank (optional).
    """

    def __init__(self, config: Optional[MultiSPINConfig] = None):
        self.config = config or MultiSPINConfig()
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.stylo_extractor = StylometricExtractor()
        self.emb_extractor = EmbeddingExtractor(self.config.embedding_model)
        self.style_extractor = None  # Lazy-loaded StyleEmbeddingDetector
        self.feature_bank = None  # Set by FeatureBank if used

    # Fallback LoRA targets for hybrid architectures (from ml_pipeline HuggingFaceQLoRAWrapper)
    _FALLBACK_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    def setup(self):
        """Load model, reference data, and feature extractors.

        Supports newer architectures (Qwen 3.5, Gemma 4, etc.) via
        trust_remote_code and LoRA target_modules fallback.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        from datasets import load_dataset
        import copy

        cfg = self.config

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
        )

        # Apply LoRA — try "all-linear" first, fall back to explicit modules
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        try:
            self.model = get_peft_model(self.model, lora_config)
        except (ValueError, RuntimeError) as e:
            logger.warning(
                "LoRA with 'all-linear' failed (%s). Using fallback modules.", e,
            )
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                target_modules=self._FALLBACK_TARGET_MODULES,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        # Reference model (frozen copy for KL)
        self.ref_model = copy.deepcopy(self.model)
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Load human reference texts
        if cfg.reference_dataset == "cnn_dailymail":
            ds = load_dataset("cnn_dailymail", "3.0.0", split="train")
            self.human_texts = ds["article"][:cfg.reference_max_samples]
        else:
            ds = load_dataset(cfg.reference_dataset, split="train")
            col = next((c for c in ["text", "article", "content"] if c in ds.column_names), ds.column_names[0])
            self.human_texts = ds[col][:cfg.reference_max_samples]

        # Pre-compute human feature vectors
        logger.info("Pre-computing human stylometric features...")
        self.human_stylo = self.stylo_extractor.extract_batch(self.human_texts[:500])
        logger.info("Pre-computing human embeddings...")
        self.human_emb = self.emb_extractor.extract_batch(self.human_texts[:500])

        # Pre-compute human style embeddings (Rivera Soto et al., 2024)
        self.human_style_emb = None
        if cfg.lambda_style > 0:
            try:
                from ...detectors.style_detector import StyleEmbeddingDetector
                self.style_extractor = StyleEmbeddingDetector(
                    model_name=cfg.style_embedding_model,
                )
                self.style_extractor._load()
                if self.style_extractor.is_available():
                    logger.info("Pre-computing human style embeddings...")
                    self.human_style_emb = self.style_extractor.get_style_embeddings_batch(
                        self.human_texts[:500]
                    )
                    logger.info("Style embeddings computed: shape %s", self.human_style_emb.shape)
                else:
                    logger.warning("Style embedding model not available, skipping style loss")
                    self.style_extractor = None
            except Exception as e:
                logger.warning("Could not load style embeddings: %s", e)
                self.style_extractor = None

        logger.info("MultiSPIN setup complete. %d human references.", len(self.human_texts))

    def _compute_spin_loss(
        self,
        human_ids: torch.Tensor,
        human_mask: torch.Tensor,
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
    ) -> torch.Tensor:
        """DPO-style SPIN loss (Chen et al., 2024).

        L_SPIN = -E[log σ(β · ((log π_θ(y_h) - log π_ref(y_h)) - (log π_θ(y_g) - log π_ref(y_g))))]
        """
        device = next(self.model.parameters()).device
        beta = float(self.config.beta_spin)

        human_ids = human_ids.to(device)
        human_mask = human_mask.to(device)
        generated_ids = generated_ids.to(device)
        generated_mask = generated_mask.to(device)

        def _seq_logprob(model, ids, mask) -> torch.Tensor:
            logits = model(input_ids=ids, attention_mask=mask).logits
            log_p = F.log_softmax(logits[:, :-1], dim=-1)
            target = ids[:, 1:]
            tok_lp = log_p.gather(2, target.unsqueeze(-1)).squeeze(-1)
            # Mask out pad and the shifted-off first position
            tok_mask = mask[:, 1:].to(tok_lp.dtype)
            return (tok_lp * tok_mask).sum(dim=-1)

        with torch.no_grad():
            ref_lp_h = _seq_logprob(self.ref_model, human_ids, human_mask)
            ref_lp_g = _seq_logprob(self.ref_model, generated_ids, generated_mask)

        curr_lp_h = _seq_logprob(self.model, human_ids, human_mask)
        curr_lp_g = _seq_logprob(self.model, generated_ids, generated_mask)

        logit_diff = beta * ((curr_lp_h - ref_lp_h) - (curr_lp_g - ref_lp_g))
        return -F.logsigmoid(logit_diff).mean()

    def _compute_feature_matching_loss(
        self,
        generated_texts: List[str],
    ) -> Dict[str, float]:
        """Compute feature-matching losses against human reference.

        Returns dict of loss components:
        - stylo_loss: MSE between stylometric features
        - emb_mmd: MMD between sentence embeddings
        - style_mmd: MMD between style embeddings (Rivera Soto et al., 2024)
        """
        cfg = self.config

        # Stylometric MSE
        gen_stylo = self.stylo_extractor.extract_batch(generated_texts)
        stylo_loss = float(np.mean((gen_stylo - self.human_stylo[:len(gen_stylo)]) ** 2))

        # Embedding MMD
        gen_emb = self.emb_extractor.extract_batch(generated_texts)
        emb_mmd = compute_mmd(gen_emb, self.human_emb[:len(gen_emb)])

        # Style embedding MMD (orthogonal authorship-style features)
        style_mmd = 0.0
        if self.style_extractor is not None and self.human_style_emb is not None:
            gen_style = self.style_extractor.get_style_embeddings_batch(generated_texts)
            style_mmd = compute_mmd(gen_style, self.human_style_emb[:len(gen_style)])

        return {
            "stylo_loss": stylo_loss,
            "emb_mmd": emb_mmd,
            "style_mmd": style_mmd,
        }

    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a list of texts with padding to the longest in the batch."""
        tok = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        )
        return tok["input_ids"], tok["attention_mask"]

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate completions for a batch of prompts with the current policy."""
        cfg = self.config
        device = next(self.model.parameters()).device
        self.model.eval()
        generated: List[str] = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=cfg.max_seq_length,
            ).to(device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=True,
                    temperature=cfg.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            text = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            generated.append(text if text.strip() else prompt)
        self.model.train()
        return generated

    def train(self):
        """Run MultiSPIN iterative training.

        Each step applies the true DPO-style SPIN loss (not CE), recomputed from a
        freshly sampled generation batch. Feature-matching losses (stylometric MSE,
        embedding MMD, style MMD) are computed per-iteration on the generation batch
        and logged as monitoring signals — they are not differentiable through the
        non-differentiable sampling step, so they do not contribute to the gradient.
        """
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.learning_rate,
        )

        prompts_pool = self.human_texts[:cfg.batch_size * 50]

        for iteration in range(cfg.num_iterations):
            logger.info("=== MultiSPIN Iteration %d/%d ===", iteration + 1, cfg.num_iterations)

            # Compute initial feature-matching monitoring signal
            init_prompts = [" ".join(t.split()[:50]) for t in prompts_pool[:cfg.batch_size]]
            generated_texts = self._generate_batch(init_prompts)
            fm_losses = self._compute_feature_matching_loss(generated_texts)
            logger.info(
                "  Init monitors | Stylo MSE: %.4f | EMB MMD: %.4f | Style MMD: %.4f",
                fm_losses["stylo_loss"], fm_losses["emb_mmd"], fm_losses["style_mmd"],
            )

            self.model.train()
            optimizer.zero_grad()
            loss_running = 0.0
            n_running = 0

            for step in range(cfg.steps_per_iteration):
                # Pick a batch of human references
                batch_start = (step * cfg.batch_size) % max(1, len(prompts_pool) - cfg.batch_size)
                human_batch = prompts_pool[batch_start : batch_start + cfg.batch_size]
                prompt_batch = [" ".join(t.split()[:50]) for t in human_batch]

                # Generate matching completions with current policy
                gen_batch = self._generate_batch(prompt_batch)

                # Tokenize human references and generations
                human_ids, human_mask = self._tokenize_batch(human_batch)
                gen_ids, gen_mask = self._tokenize_batch(gen_batch)

                # True SPIN loss
                spin_loss = self._compute_spin_loss(human_ids, human_mask, gen_ids, gen_mask)
                total_loss = cfg.lambda_spin * spin_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0,
                )

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                loss_running += float(spin_loss.item())
                n_running += 1

                # Refresh monitoring feature-matching losses periodically
                if (step + 1) % 100 == 0:
                    fm_losses = self._compute_feature_matching_loss(gen_batch)
                    logger.info(
                        "  Step %d/%d | SPIN loss (avg %d): %.4f | Stylo: %.4f | EMB MMD: %.4f | Style MMD: %.4f",
                        step + 1, cfg.steps_per_iteration, n_running,
                        loss_running / max(n_running, 1),
                        fm_losses["stylo_loss"],
                        fm_losses["emb_mmd"],
                        fm_losses["style_mmd"],
                    )
                    loss_running = 0.0
                    n_running = 0

            # Save iteration checkpoint
            iter_dir = os.path.join(cfg.output_dir, f"iteration_{iteration + 1}")
            self.model.save_pretrained(iter_dir)
            self.tokenizer.save_pretrained(iter_dir)

            # Update feature bank if attached
            if self.feature_bank is not None:
                self.feature_bank.update(generated_texts, self.human_texts[:len(generated_texts)])

            logger.info("Iteration %d complete. Checkpoint saved to %s", iteration + 1, iter_dir)

        logger.info("MultiSPIN training complete.")
