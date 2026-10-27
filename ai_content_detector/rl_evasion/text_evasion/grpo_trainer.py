"""GRPO-based text evasion trainer.

Implements GRPO + LoRA on an open LLM, using detector scores as reward
signals. This is the core text-evasion training loop.

References:
    - Shao et al., DeepSeekMath (arXiv:2402.03300), which introduces GRPO.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..config import TextEvasionConfig
from .rewards import CompositeReward, DetectorReward, SemanticSimilarityReward, build_detector_reward_from_name

from .grpo_optimization import GRPOOptimization

logger = logging.getLogger(__name__)


class GRPOTextEvasionTrainer(GRPOOptimization):
    """Train an LLM via GRPO to produce text that evades AI detectors.

    The generator is fine-tuned with LoRA. The reward is a composite of:
    - Evasion: 1 - detector_score (averaged across an ensemble)
    - Semantic preservation: cosine similarity of sentence embeddings
    - Quality: length ratio penalty

    Uses HuggingFace TRL's GRPOTrainer under the hood.
    """

    def __init__(self, config: Optional[TextEvasionConfig] = None):
        self.config = config or TextEvasionConfig()
        self.model = None
        self.tokenizer = None
        self.reward_fn = None
        self._trainer = None

    def setup(self):
        """Load model, tokenizer, LoRA, and reward functions."""
        from ..config import seed_everything
        seed_everything(self.config.seed)
        logger.info("Setting up GRPO text evasion trainer...")
        self._setup_model()
        self._setup_rewards()
        self._setup_dataset()
        logger.info("Setup complete.")

    # Fallback LoRA targets for hybrid MoE/SSM architectures (from ml_pipeline HuggingFaceQLoRAWrapper)
    _FALLBACK_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    def _setup_model(self):
        """Load base model with LoRA adapters.

        Supports newer architectures (Qwen 3.5, Gemma 4, etc.) via
        trust_remote_code and LoRA target_modules fallback, following
        the pattern from ml_pipeline.pipelines_torch.models.HuggingFaceQLoRAWrapper.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        cfg = self.config

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.generator_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load in BF16/FP16 for efficiency
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.generator_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # Apply LoRA — try "all-linear" first, fall back to explicit module
        # names for hybrid architectures (pattern from HuggingFaceQLoRAWrapper)
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        try:
            self.model = get_peft_model(self.model, lora_config)
        except (ValueError, RuntimeError) as e:
            logger.warning(
                "LoRA with target_modules='all-linear' failed (%s). "
                "Retrying with fallback modules: %s", e, self._FALLBACK_TARGET_MODULES,
            )
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=self._FALLBACK_TARGET_MODULES,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info("Model loaded. Trainable: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    def _setup_rewards(self):
        """Build the composite reward function."""
        cfg = self.config
        device = "cuda" if torch.cuda.is_available() else "cpu"

        detector_rewards = []
        for name in cfg.detector_names:
            try:
                dr = build_detector_reward_from_name(name, device=device)
                detector_rewards.append(dr)
                logger.info("Loaded detector reward: %s", name)
            except Exception as e:
                logger.warning("Could not load detector %s: %s", name, e)
        if not detector_rewards:
            raise RuntimeError(
                f"No requested text detector could be loaded: {cfg.detector_names}"
            )

        semantic_reward = SemanticSimilarityReward(
            model_name=cfg.embedding_model,
            device=device,
        )

        self.reward_fn = CompositeReward(
            detector_rewards=detector_rewards,
            semantic_reward=semantic_reward,
            weights={
                "evasion": cfg.reward_evasion_weight,
                "semantic": cfg.reward_semantic_weight,
                "quality": cfg.reward_quality_weight,
            },
        )

    def _setup_dataset(self):
        """Load reference corpus for prompts."""
        from datasets import load_dataset

        cfg = self.config
        logger.info("Loading reference dataset: %s", cfg.reference_dataset)

        if cfg.reference_dataset == "cnn_dailymail":
            ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=cfg.reference_split)
            texts = ds["article"][:cfg.reference_max_samples]
        else:
            ds = load_dataset(cfg.reference_dataset, split=cfg.reference_split)
            # Try common text column names
            text_col = next(
                (c for c in ["text", "article", "content", "document"] if c in ds.column_names),
                ds.column_names[0],
            )
            texts = ds[text_col][:cfg.reference_max_samples]

        # Create prompts: take first ~50 words as prompt, rest as reference
        self.prompts = []
        self.references = []
        for text in texts:
            words = text.split()
            if len(words) < 70:
                continue
            prompt = " ".join(words[:50])
            reference = " ".join(words[50:])
            self.prompts.append(prompt)
            self.references.append(reference)

        if len(self.prompts) < 5:
            raise ValueError("GRPO requires at least five prompt/reference pairs")
        from ml_pipeline.utils.data_split import grouped_holdout_indices
        n_eval = max(1, int(round(cfg.evaluation_fraction * len(self.prompts))))
        train_indices, eval_indices = grouped_holdout_indices(self.prompts, n_eval, seed=cfg.seed)
        all_prompts, all_references = self.prompts, self.references
        self.eval_prompts = [all_prompts[index] for index in eval_indices]
        self.eval_references = [all_references[index] for index in eval_indices]
        self.prompts = [all_prompts[index] for index in train_indices]
        self.references = [all_references[index] for index in train_indices]

        logger.info("Prepared %d prompt-reference pairs.", len(self.prompts))

    def compute_reward(
        self,
        prompts: List[str],
        completions: List[str],
        references: Optional[List[str]] = None,
    ) -> List[float]:
        """Compute rewards for a batch of completions.

        For GRPO, this is called on each group of generations per prompt.
        """
        rewards = []
        sources = references if references is not None else prompts
        if hasattr(self.reward_fn, "batch"):
            return [result["total"] for result in self.reward_fn.batch(sources, completions)]
        for source, completion in zip(sources, completions):
            result = self.reward_fn(source, completion)
            rewards.append(result["total"])
        return rewards

    def train(self, max_steps: Optional[int] = None):
        from ..checkpointing import StopRequest
        with StopRequest() as self._stop_request:
            return self._train(max_steps=max_steps)

    def _train(self, max_steps: Optional[int] = None):
        """Run GRPO training loop.

        Uses TRL's GRPOTrainer if available, otherwise falls back to a
        manual GRPO implementation.
        """
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        self.training_history = []
        with open(os.path.join(cfg.output_dir, "training_metadata.json"), "w") as metadata_file:
            json.dump({
                "method": "grpo" if not cfg.allow_reinforce_fallback else "grpo_or_reinforce_fallback",
                "detector_names": cfg.detector_names,
                "seed": cfg.seed,
                "training_samples": len(self.prompts),
                "evaluation_samples": len(self.eval_prompts),
            }, metadata_file, indent=2)

        try:
            self._train_with_trl(max_steps=max_steps)
        except ImportError as error:
            if not cfg.allow_reinforce_fallback:
                raise RuntimeError(
                    "TRL is required for GRPO. Set allow_reinforce_fallback=True only "
                    "to run the separately named group-normalized REINFORCE baseline."
                ) from error
            logger.warning("TRL unavailable; running group-normalized REINFORCE baseline")
            self._train_group_normalized_reinforce(max_steps=max_steps)

    def _load_restart_context(self, path):
        from ..checkpointing import load_checkpoint
        context = load_checkpoint(path)
        if context is not None and TextEvasionConfig(**context["config"]) != self.config:
            raise ValueError(f"Restart configuration differs from {path}")
        return context


    def _copy_training_metadata(self, destination: str) -> None:
        import shutil
        source = os.path.join(self.config.output_dir, "training_metadata.json")
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(destination, "training_metadata.json"))

    def evaluate(
        self,
        num_samples: int = 100,
        prompt_indices: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Evaluate the trained model's evasion capability.

        If ``prompt_indices`` is given, evaluation runs only on those prompts
        (used by the arms race to evaluate on a held-out split that is disjoint
        from the prompts the defender was retrained on). Otherwise it falls back
        to the first ``num_samples`` prompts.
        """
        from .evaluate import evaluate_text_evasion

        device = next(self.model.parameters()).device
        self.model.eval()

        from dataclasses import asdict
        from ..checkpointing import (
            CheckpointPause, StopRequest, atomic_save, evaluation_state, rng_state,
        )

        if prompt_indices:
            eval_indices = list(prompt_indices)[:num_samples]
            eval_prompts = self.prompts
            eval_references = self.references
        else:
            eval_prompts = getattr(self, "eval_prompts", self.prompts)
            eval_references = getattr(self, "eval_references", self.references)
            eval_indices = list(range(min(num_samples, len(eval_prompts))))
        pairs = [(eval_prompts[i], eval_references[i] or eval_prompts[i]) for i in eval_indices]
        path = Path(self.config.output_dir) / "evaluation_restart.pt"
        state = evaluation_state(path, {"config": asdict(self.config), "pairs": pairs}, self.model)

        def save():
            state["rng"] = rng_state()
            atomic_save(state, path)

        with StopRequest() as stop:
            for prompt, source in pairs[len(state["generated"]):]:
                if stop.requested:
                    raise CheckpointPause("GRPO evaluation saved")
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                        max_length=self.config.max_seq_length).to(device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature, top_p=self.config.top_p, do_sample=True,
                    )
                state["generated"].append(self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                ))
                save()
            for index in range(len(state["scores"]), len(pairs)):
                if stop.requested:
                    raise CheckpointPause("GRPO evaluation saved")
                state["scores"].append(self.reward_fn(pairs[index][1], state["generated"][index]))
                save()
            if stop.requested:
                raise CheckpointPause("GRPO evaluation saved")
        scores = iter(state["scores"])
        return evaluate_text_evasion(
            state["generated"], [source for _, source in pairs], lambda source, generated: next(scores),
        )
