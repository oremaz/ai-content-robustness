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

import numpy as np
import torch
from datasets import Dataset

from ..config import TextEvasionConfig
from .rewards import CompositeReward, DetectorReward, SemanticSimilarityReward, build_detector_reward_from_name

logger = logging.getLogger(__name__)


class GRPOTextEvasionTrainer:
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

    def _train_with_trl(self, max_steps: Optional[int] = None):
        """Train using HuggingFace TRL's GRPOTrainer."""
        from trl import GRPOConfig, GRPOTrainer

        cfg = self.config

        candidates = getattr(self, "microbatch_candidates", [])
        if candidates:
            from .batch_layout import restore_layout
            restore_layout(self, candidates)

        # Prepare dataset as HF Dataset
        train_dataset = Dataset.from_dict({"prompt": self.prompts, "reference": self.references})

        # Reward function for TRL
        def reward_fn(completions: list, **kwargs) -> list:
            prompts_batch = kwargs.get("prompts", [""] * len(completions))
            references_batch = kwargs.get("reference", prompts_batch)
            texts = [completion[0]["content"] if isinstance(completion, list) else str(completion)
                     for completion in completions]
            return [torch.tensor(value) for value in self.compute_reward(prompts_batch, texts, references_batch)]

        grpo_config = GRPOConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            lr_scheduler_kwargs=cfg.lr_scheduler_kwargs,
            warmup_steps=cfg.warmup_steps,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            num_generations=cfg.grpo_num_generations,
            beta=cfg.grpo_beta,
            num_iterations=1,
            max_completion_length=cfg.max_new_tokens,
            seed=cfg.seed,
            max_steps=max_steps if max_steps is not None else -1,
            save_total_limit=2,
        )
        # TRL renamed ``tokenizer`` → ``processing_class`` around 0.14. Pass both if
        # accepted, else fall back, so we don't break on either version.
        try:
            trainer = GRPOTrainer(
                model=self.model,
                args=grpo_config,
                train_dataset=train_dataset,
                reward_funcs=reward_fn,
                processing_class=self.tokenizer,
            )
        except TypeError:
            trainer = GRPOTrainer(
                model=self.model,
                args=grpo_config,
                train_dataset=train_dataset,
                reward_funcs=reward_fn,
                tokenizer=self.tokenizer,
            )

        if candidates:
            from .batch_layout import select_layout
            select_layout(self, trainer, candidates)
        with open(os.path.join(cfg.output_dir, "trl_training_arguments.json"), "w") as handle:
            json.dump(grpo_config.to_dict(), handle, indent=2, default=str)

        from .schedules import configure_scheduler
        configure_scheduler(trainer, cfg)
        from .reference import configure_reference_restart
        configure_reference_restart(trainer, cfg)

        from dataclasses import asdict
        from transformers import TrainerCallback
        from ..checkpointing import (
            CheckpointClock, CheckpointPause, atomic_save, atomic_transformers_save,
            load_checkpoint, map_tensors, rng_state, restore_rng,
        )
        context_path = Path(cfg.output_dir) / "restart_context.pt"
        context = self._load_restart_context(context_path)
        if context is None:
            atomic_save({
                "config": asdict(cfg), "prompts": self.prompts, "references": self.references,
                "eval_prompts": self.eval_prompts, "eval_references": self.eval_references,
                "max_steps": max_steps,
            }, context_path)
        elif (context["prompts"] != self.prompts or context["references"] != self.references
              or context["eval_prompts"] != self.eval_prompts
              or context["eval_references"] != self.eval_references
              or context["max_steps"] != max_steps):
            raise ValueError("GRPO dataset split or step budget differs from restart context")
        stop_request = self._stop_request
        clock = CheckpointClock()

        native_save = trainer._save_checkpoint

        def save_checkpoint(model, trial=None):
            atomic_transformers_save(trainer, native_save, model, trial, {
                "step": getattr(trainer, "_step", 0),
                "buffered_inputs": map_tensors(
                    getattr(trainer, "_buffered_inputs", None), lambda tensor: tensor.detach().cpu()
                ),
            })
            if trainer.state.global_step == 500:
                import shutil
                destination = Path(cfg.output_dir) / "milestones/step_500"
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(".saving_step_500")
                    shutil.rmtree(temporary, ignore_errors=True)
                    shutil.copytree(Path(cfg.output_dir) / "checkpoint-500", temporary)
                    shutil.copy2(context_path, temporary / "research_restart_context.pt")
                    (temporary / "original_training_arguments.json").write_text(
                        json.dumps(grpo_config.to_dict(), indent=2, default=str) + "\n")
                    os.replace(temporary, destination)
            clock.saved()

        trainer._save_checkpoint = save_checkpoint
        diagnostic = getattr(self, "log_training_diagnostic", None)

        class RestartCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                if resume is None:
                    trainer._save_checkpoint(trainer.model, trial=None)
                    if diagnostic is not None:
                        diagnostic(0)
                return control

            def on_step_end(self, args, state, control, **kwargs):
                if clock.due() or stop_request.requested:
                    control.should_save = True
                if diagnostic is not None and state.global_step % 50 == 0:
                    control.should_save = True
                if stop_request.requested:
                    control.should_training_stop = True
                return control

            def on_save(self, args, state, control, **kwargs):
                if diagnostic is not None and state.global_step > 0 and state.global_step % 50 == 0 and not stop_request.requested:
                    diagnostic(state.global_step)
                return control

            def on_train_end(self, args, state, control, **kwargs):
                if diagnostic is not None and state.global_step > 0 and not stop_request.requested:
                    trainer._save_checkpoint(trainer.model, trial=None)
                    diagnostic(state.global_step)
                return control

        trainer.add_callback(RestartCallback())
        checkpoints = sorted(
            (path for path in Path(cfg.output_dir).glob("checkpoint-*")
             if path.is_dir() and (path / "restart_complete.pt").is_file()),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        resume = str(checkpoints[-1]) if checkpoints else None
        if resume is not None:
            internal = load_checkpoint(Path(resume) / "restart_complete.pt")
            trainer._step = internal["step"]
            trainer._buffered_inputs = map_tensors(
                internal["buffered_inputs"], lambda tensor: tensor.to(next(self.model.parameters()).device)
            )
        self._trainer = trainer
        if resume is not None and diagnostic is not None and not context.get("completed"):
            resumed_step = int(Path(resume).name.split("-")[-1])
            if resumed_step % 50 == 0:
                trainer._load_from_checkpoint(resume, self.model)
                diagnostic(resumed_step)
                if stop_request.requested:
                    raise CheckpointPause("GRPO diagnostic progress retained before training continuation")
        if context is not None and context.get("completed"):
            if resume is None:
                raise ValueError("Completed GRPO restart is missing its full checkpoint")
            if max_steps is not None and int(Path(resume).name.split("-")[-1]) != max_steps:
                raise ValueError("Completed GRPO restart does not match the fixed update budget")
            trainer._load_from_checkpoint(resume, self.model)
            self.training_history = json.loads((Path(resume) / "trainer_state.json").read_text())["log_history"]
            restore_rng(context["rng"])
            return
        trainer.train(resume_from_checkpoint=resume)
        self.training_history = trainer.state.log_history
        trainer._save_checkpoint(trainer.model, trial=None)
        if stop_request.requested:
            raise CheckpointPause("GRPO optimizer, scheduler, RNG and sampler progress saved")
        if max_steps is not None and trainer.state.global_step != max_steps:
            raise ValueError("GRPO ended before completing its fixed update budget")
        trainer.save_model(os.path.join(cfg.output_dir, "final"))
        self.tokenizer.save_pretrained(os.path.join(cfg.output_dir, "final"))
        self._copy_training_metadata(os.path.join(cfg.output_dir, "final"))
        context = self._load_restart_context(context_path)
        context["completed"] = True
        context["rng"] = rng_state()
        atomic_save(context, context_path)
        logger.info("TRL GRPO training complete. Model saved to %s", cfg.output_dir)

    def _train_group_normalized_reinforce(self, max_steps: Optional[int] = None):
        """Manual group-normalized REINFORCE baseline without PPO clipping.

        Reference-faithful GRPO (DeepSeekMath 2024):
            1. For each prompt, sample K completions under the current policy.
            2. Reward each completion, then compute the group-normalized advantage
               A_i = (r_i - mean(r)) / (std(r) + eps).
            3. Loss per sample:
                  L_i = -A_i * sum_t log π_θ(y_t | ·)
                        + β * KL_token(π_θ ‖ π_ref)
               using the k2 KL estimator KL ≈ 0.5 * (log π_θ - log π_ref)^2,
               and a frozen reference obtained by disabling the LoRA adapter.
            4. Average across the group, gradient-clip, step every
               ``gradient_accumulation_steps`` optimizer updates.
        """
        cfg = self.config
        if cfg.lr_scheduler_type != "linear" or cfg.warmup_steps or cfg.lr_scheduler_kwargs or cfg.grpo_beta:
            raise ValueError("Configured learning-rate schedules require TRL GRPO")
        device = next(self.model.parameters()).device
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.learning_rate,
        )

        num_steps = len(self.prompts) * cfg.num_train_epochs
        log_interval = cfg.logging_steps
        save_interval = cfg.save_steps
        min_out = cfg.grpo_min_output_tokens

        step = 0
        all_rewards: List[float] = []

        def _ref_logprobs(input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            """Per-token log-probs of `input_ids` under the frozen reference.

            If the model is a PeftModel, disable the LoRA adapter so the base weights
            act as the reference; otherwise fall back to a frozen forward pass.
            """
            disable = getattr(self.model, "disable_adapter", None)
            ctx = disable() if callable(disable) else torch.no_grad()
            with ctx:
                with torch.no_grad():
                    logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
            log_p = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = input_ids[:, 1:]
            return log_p.gather(2, tgt.unsqueeze(-1)).squeeze(-1)

        for epoch in range(cfg.num_train_epochs):
            indices = np.random.permutation(len(self.prompts))

            for idx in indices:
                if max_steps is not None and step >= max_steps:
                    break
                prompt = self.prompts[idx]
                inputs = self.tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=cfg.max_seq_length,
                ).to(device)
                prompt_len = inputs["input_ids"].shape[1]

                # --- Sample K completions (no grad) --------------------------
                self.model.eval()
                completions: List[str] = []
                seqs: List[torch.Tensor] = []
                for _ in range(cfg.grpo_num_generations):
                    with torch.no_grad():
                        out = self.model.generate(
                            **inputs,
                            max_new_tokens=cfg.max_new_tokens,
                            temperature=cfg.temperature,
                            top_p=cfg.top_p,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    seqs.append(out[0])
                    completions.append(
                        self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
                    )

                # --- Score completions ---------------------------------------
                rewards_np = np.array([
                    self.reward_fn(self.references[idx], comp)["total"] for comp in completions
                ], dtype=np.float32)
                all_rewards.extend(rewards_np.tolist())

                # --- Group-normalized advantages ------------------------------
                std = float(rewards_np.std())
                if std > 1e-8:
                    advantages = (rewards_np - rewards_np.mean()) / std
                else:
                    advantages = np.zeros_like(rewards_np)

                # --- Reject degenerate completions before computing gradient --
                keep_mask = np.array([
                    len(c.split()) >= min_out for c in completions
                ], dtype=bool)
                if not keep_mask.any():
                    step += 1
                    continue

                # --- Compute policy-gradient + KL loss per sample ------------
                self.model.train()
                batch_loss = torch.zeros((), device=device, dtype=torch.float32)
                for i, seq in enumerate(seqs):
                    if not keep_mask[i]:
                        continue
                    seq = seq.to(device).unsqueeze(0)
                    attn = torch.ones_like(seq)

                    out = self.model(input_ids=seq, attention_mask=attn)
                    log_p = torch.log_softmax(out.logits[:, :-1], dim=-1)
                    tgt = seq[:, 1:]
                    curr_tok_lp = log_p.gather(2, tgt.unsqueeze(-1)).squeeze(-1)

                    ref_tok_lp = _ref_logprobs(seq, attn)

                    # Only count the completion tokens (after the prompt).
                    comp_mask = torch.zeros_like(curr_tok_lp)
                    comp_mask[:, prompt_len - 1 :] = 1.0
                    n_comp = comp_mask.sum().clamp(min=1.0)

                    # Policy gradient: maximize A_i * log π_θ over completion tokens.
                    pg = -float(advantages[i]) * (curr_tok_lp * comp_mask).sum() / n_comp

                    # k2 KL estimator (unbiased in expectation, low variance).
                    log_ratio = (curr_tok_lp - ref_tok_lp) * comp_mask
                    kl = 0.5 * (log_ratio.pow(2).sum() / n_comp)

                    batch_loss = batch_loss + pg + cfg.grpo_kl_coeff * kl

                batch_loss = batch_loss / float(keep_mask.sum())
                (batch_loss / cfg.gradient_accumulation_steps).backward()

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        cfg.grpo_max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                self.training_history.append({"step": step, "loss": float(batch_loss.item()),
                                              "reward": float(rewards_np.mean())})

                if step % log_interval == 0:
                    avg_reward = np.mean(all_rewards[-100:]) if all_rewards else 0.0
                    logger.info(
                        "Step %d/%d | Avg Reward (last 100): %.4f | Best: %.4f | Kept: %d/%d",
                        step, num_steps, avg_reward, float(rewards_np.max()),
                        int(keep_mask.sum()), int(len(keep_mask)),
                    )

                if step % save_interval == 0:
                    ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint-{step}")
                    self.model.save_pretrained(ckpt_dir)
                    self.tokenizer.save_pretrained(ckpt_dir)

            if max_steps is not None and step >= max_steps:
                break

        if step % cfg.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                cfg.grpo_max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

        # Save final
        final_dir = os.path.join(cfg.output_dir, "final")
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        self._copy_training_metadata(final_dir)
        logger.info("Manual GRPO training complete. Model saved to %s", final_dir)

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
