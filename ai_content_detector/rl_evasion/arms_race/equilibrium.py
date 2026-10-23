"""Multi-round adversarial attacker/defender experiments.

Alternates between an attacker update and adaptive detector retraining. Metrics
measure both updates on a fixed, untouched evaluation split. No Nash-equilibrium
claim is made because the experiment does not compute exact best responses.

Reference: todo/rl_evasion_research_directions_v2.md §Thread 2 —
"The equilibrium experiment (most important experiment in the project)"

Protocol:
    Round 0: initial generator + initial detector ensemble
    Round r → r+1:
        1. Attacker: fine-tune generator against current detectors (GRPO/MultiSPIN)
        2. Defender: retrain a classifier on new generator outputs
        3. Evaluate: TPR@1%FPR, AUROC, attack success rate
    Run for N rounds and report paired adaptation curves.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path
import logging
import os
from typing import Dict, List, Optional

import numpy as np

from ..config import ArmsRaceConfig, TextEvasionConfig
from ..checkpointing import (
    CheckpointPause, StopRequest, atomic_save, load_checkpoint, model_state,
    restore_model, restore_rng, rng_state,
)

logger = logging.getLogger(__name__)


class ArmsRaceExperiment:
    """Run the multi-round attacker/defender arms race."""

    def __init__(self, config: Optional[ArmsRaceConfig] = None):
        self.config = config or ArmsRaceConfig()
        self.history: List[Dict] = []  # per-round metrics
        self.attacker = None
        self.defender = None
        self.train_prompts: List[str] = []
        self.train_references: List[str] = []
        self.eval_prompts: List[str] = []
        self.eval_references: List[str] = []
        self._restart = None
        self._generation_context = "generation"

    def setup(self):
        """Initialize attacker and defender."""
        cfg = self.config

        if cfg.modality == "text":
            self._setup_text()
        elif cfg.modality == "image":
            self._setup_image()
        else:
            raise ValueError(f"Unknown modality: {cfg.modality}")

        os.makedirs(cfg.output_dir, exist_ok=True)
        logger.info("Arms race experiment setup complete. Modality: %s, rounds: %d", cfg.modality, cfg.num_rounds)

    def _setup_text(self):
        """Set up the text attacker and adaptive classifier defender."""
        from ..text_evasion.grpo_trainer import GRPOTextEvasionTrainer
        from .radar_defender import AdaptiveClassifierDefender

        attacker_config = self.config.attacker_config or TextEvasionConfig(
            num_train_epochs=1,
            output_dir=os.path.join(self.config.output_dir, "attacker"),
        )
        attacker_config.output_dir = os.path.join(self.config.output_dir, "attacker")

        self.attacker = GRPOTextEvasionTrainer(config=attacker_config)
        self.attacker.setup()
        self._partition_prompts()

        self.defender = AdaptiveClassifierDefender(
            model_name=self.config.defender_model,
            output_dir=os.path.join(self.config.output_dir, "defender"),
            retrain_samples=self.config.defender_retrain_samples,
            epochs=self.config.defender_epochs,
            lr=self.config.defender_lr,
            seed=self.config.seed,
        )
        self.defender.setup()
        self._restore_restart()
        if self._restart["phase"] == "warmup":
            self._warmup_defender()
            self._restart["phase"] = "before_attack"
            self._save_restart()

    def _warmup_defender(self) -> None:
        n = min(self.config.defender_retrain_samples // 2, len(self.train_prompts))
        if n < 2:
            raise ValueError("Defender warm-up requires at least two training prompt/reference pairs")
        self._generation_context = "warmup"
        ai_texts = self._generate(self.train_prompts[:n], seed=self.config.seed + 1)
        metrics = self.defender.retrain(
            ai_texts=ai_texts,
            human_texts=self.train_references[:n],
            warmup=True,
        )
        self._update_attacker_detectors()
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(os.path.join(self.config.output_dir, "defender_warmup.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info("Defender warm-up complete: %s", metrics)

    def _partition_prompts(self, eval_frac: float = 0.2) -> None:
        """Create an immutable split and remove evaluation prompts from training."""
        if getattr(self.attacker, "eval_prompts", None):
            self.train_prompts = list(self.attacker.prompts)
            self.train_references = list(self.attacker.references)
            self.eval_prompts = list(self.attacker.eval_prompts)
            self.eval_references = list(self.attacker.eval_references)
            return
        n = len(self.attacker.prompts)
        if n < 5:
            raise ValueError("Arms-race evaluation requires at least five prompt/reference pairs")
        prompts = list(self.attacker.prompts)
        references = list(self.attacker.references)
        rng = np.random.default_rng(self.config.seed)
        perm = rng.permutation(n)
        n_eval = max(1, int(eval_frac * n))
        eval_idx = sorted(perm[:n_eval].tolist())
        train_idx = sorted(perm[n_eval:].tolist())
        self.train_prompts = [prompts[i] for i in train_idx]
        self.train_references = [references[i] for i in train_idx]
        self.eval_prompts = [prompts[i] for i in eval_idx]
        self.eval_references = [references[i] for i in eval_idx]
        self.attacker.prompts = self.train_prompts
        self.attacker.references = self.train_references
        logger.info(
            "Arms-race prompt split: %d train / %d immutable eval",
            len(self.train_prompts), len(self.eval_prompts),
        )

    def _setup_image(self):
        raise NotImplementedError(
            "Image arms-race mode requires an adaptive image defender and a common "
            "held-out payoff evaluation; neither is implemented yet"
        )

    def _restore_restart(self):
        path = Path(self.config.output_dir) / "race_restart.pt"
        state = load_checkpoint(path, asdict(self.config))
        manifest = {
            "train_prompts": self.train_prompts, "train_references": self.train_references,
            "eval_prompts": self.eval_prompts, "eval_references": self.eval_references,
        }
        if state is None:
            self._restart = {
                "config": asdict(self.config), "manifest": manifest,
                "round": 0, "phase": "warmup", "metrics": {}, "history": [],
            }
            self._save_restart()
        else:
            if state["manifest"] != manifest:
                raise ValueError("Arms-race restart prompt/reference manifest differs")
            restore_model(self.attacker.model, state["attacker"])
            restore_model(self.defender.model, state["defender"])
            self.defender._round = state["defender_round"]
            self.history = state["history"]
            self._restart = state
            self._update_attacker_detectors()
            restore_rng(state["rng"])

    def _save_restart(self):
        self._restart.update(
            attacker=model_state(self.attacker.model),
            defender=model_state(self.defender.model),
            defender_round=self.defender._round,
            rng=rng_state(), history=self.history,
        )
        atomic_save(self._restart, Path(self.config.output_dir) / "race_restart.pt")

    def _generate(self, prompts: List[str], seed: int) -> List[str]:
        import torch

        device = next(self.attacker.model.parameters()).device
        self.attacker.model.eval()
        generator = torch.Generator(device=device).manual_seed(seed)
        signature = {
            "prompts": prompts, "seed": seed, "context": self._generation_context,
            "model": self.attacker.config.generator_model,
            "max_seq_length": self.attacker.config.max_seq_length,
            "max_new_tokens": self.attacker.config.max_new_tokens,
            "temperature": self.attacker.config.temperature,
            "top_p": self.attacker.config.top_p,
        }
        identity = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        path = Path(self.config.output_dir) / "generation" / f"{identity}.pt"
        state = load_checkpoint(path, signature)
        if state is None:
            state = {"config": signature, "texts": [], "generator": generator.get_state()}
            atomic_save(state, path)
        generator.set_state(state["generator"])
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with StopRequest() as stop:
            for prompt in prompts[len(state["texts"]):]:
                if stop.requested:
                    raise CheckpointPause("Arms-race generation restart state saved")
                inputs = self.attacker.tokenizer(
                    prompt, return_tensors="pt", truncation=True,
                    max_length=self.attacker.config.max_seq_length,
                ).to(device)
                # HF generate samples through the device's default generator.
                # Forking preserves the caller RNG and permits exact sample restart.
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    if device.type == "cuda":
                        torch.cuda.set_rng_state(generator.get_state(), device)
                    else:
                        torch.set_rng_state(generator.get_state())
                    generated = self.attacker.model.generate(
                        **inputs, max_new_tokens=self.attacker.config.max_new_tokens,
                        temperature=self.attacker.config.temperature,
                        top_p=self.attacker.config.top_p, do_sample=True,
                    )
                    generator.set_state(torch.cuda.get_rng_state(device)
                                        if device.type == "cuda" else torch.get_rng_state())
                state["texts"].append(self.attacker.tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                ))
                state["generator"] = generator.get_state()
                atomic_save(state, path)
        return state["texts"]

    def _evaluate_defender(self, ai_texts: List[str]) -> Dict[str, float]:
        from ..text_evasion.evaluate import compute_auroc

        human_texts = self.eval_references[:len(ai_texts)]
        if len(human_texts) != len(ai_texts):
            raise ValueError("Held-out AI and human evaluation samples must be aligned")
        ai_scores = np.asarray(self.defender.predict_batch(ai_texts), dtype=np.float64)
        human_scores = np.asarray(self.defender.predict_batch(human_texts), dtype=np.float64)
        scores = np.concatenate([ai_scores, human_scores])
        labels = [1] * len(ai_scores) + [0] * len(human_scores)
        predictions = scores >= 0.5
        return {
            "accuracy": float(np.mean(predictions == np.asarray(labels))),
            "auroc": compute_auroc(scores.tolist(), labels),
            "ai_detection_rate": float(np.mean(ai_scores >= 0.5)),
            "attack_success_rate": float(np.mean(ai_scores < 0.5)),
            "num_ai": len(ai_scores),
            "num_human": len(human_scores),
        }

    @staticmethod
    def _compute_update_effects(
        before_attack: Dict[str, float],
        after_attack: Dict[str, float],
        after_defense: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute paired update effects from one held-out evaluation protocol."""
        return {
            "attacker_success_change": float(
                after_attack["attack_success_rate"] - before_attack["attack_success_rate"]
            ),
            "defender_accuracy_change": float(
                after_defense["accuracy"] - after_attack["accuracy"]
            ),
            "defender_auroc_change": float(
                after_defense["auroc"] - after_attack["auroc"]
            ),
        }

    def run(self):
        """Run paired updates, resuming a completed phase instead of replaying it."""
        cfg = self.config
        if self._restart is None:
            self._restore_restart()
        state = self._restart
        with StopRequest() as stop:
            while state["round"] < cfg.num_rounds:
                if stop.requested:
                    raise CheckpointPause("Arms-race phase restart state saved")
                index = state["round"]
                metrics = state["metrics"]
                metrics.setdefault("round", index + 1)
                metrics.setdefault("eval_samples", min(cfg.eval_prompts, len(self.eval_prompts)))
                evaluation = self.eval_prompts[:cfg.eval_prompts]
                seed = cfg.seed + 10_000 * index
                phase = state["phase"]
                self._generation_context = f"round_{index + 1}_{phase}"

                if phase == "before_attack":
                    texts = self._generate(evaluation, seed=seed)
                    metrics["before_attacker_update"] = self._evaluate_defender(texts)
                    state["phase"] = "attacker"
                elif phase == "attacker":
                    self.attacker.config.output_dir = os.path.join(
                        cfg.output_dir, "attacker", f"round_{index + 1}",
                    )
                    self.attacker.train(max_steps=cfg.attacker_steps_per_round)
                    state["phase"] = "after_attack"
                elif phase == "after_attack":
                    state["post_attack_texts"] = self._generate(evaluation, seed=seed)
                    metrics["after_attacker_update"] = self._evaluate_defender(state["post_attack_texts"])
                    state["phase"] = "generate_defender"
                elif phase == "generate_defender":
                    count = min(cfg.defender_retrain_samples // 2, len(self.train_prompts))
                    state["training_texts"] = self._generate(self.train_prompts[:count], seed=seed + 2)
                    state["phase"] = "defender"
                elif phase == "defender":
                    self.defender._round = index
                    metrics["defender_training"] = self.defender.retrain(
                        ai_texts=state["training_texts"],
                        human_texts=self.train_references[:len(state["training_texts"])],
                    )
                    self._update_attacker_detectors()
                    state["phase"] = "after_defense"
                elif phase == "after_defense":
                    metrics["after_defender_update"] = self._evaluate_defender(state["post_attack_texts"])
                    metrics["update_effects"] = self._compute_update_effects(
                        metrics["before_attacker_update"], metrics["after_attacker_update"],
                        metrics["after_defender_update"],
                    )
                    self.history.append(dict(metrics))
                    state["round"] += 1
                    state["metrics"] = {}
                    state.pop("post_attack_texts")
                    state.pop("training_texts")
                    state["phase"] = "before_attack"
                else:
                    raise ValueError(f"Unknown arms-race restart phase: {phase}")
                self._save_restart()
                self._save_history()
        self._print_summary()

    def _update_attacker_detectors(self):
        """Replace one of the attacker's detector rewards with the retrained defender."""
        if self.defender is None or self.attacker is None:
            return

        from ..text_evasion.rewards import DetectorReward

        def defender_detect(text: str) -> float:
            return self.defender.predict(text)

        new_reward = DetectorReward(defender_detect, name="adaptive_defender")

        # Find and replace existing radar reward, or append
        for i, dr in enumerate(self.attacker.reward_fn.detector_rewards):
            if dr.name == "adaptive_defender":
                self.attacker.reward_fn.detector_rewards[i] = new_reward
                return
        self.attacker.reward_fn.detector_rewards.append(new_reward)

    def _save_history(self):
        """Save experiment history to JSON."""
        path = os.path.join(self.config.output_dir, "arms_race_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def _print_summary(self):
        """Print paired attacker and defender update effects."""
        print(f"\n{'=' * 95}")
        print("  ARMS RACE UPDATE SUMMARY")
        print(f"{'=' * 95}")
        print(f"  {'Round':>5}  {'Attack Success':>15}  {'Attacker Delta':>15}  {'Defender AUROC':>15}  {'Defender Delta':>15}")
        print(f"  {'-'*5}  {'-'*15}  {'-'*15}  {'-'*15}  {'-'*10}")

        for entry in self.history:
            r = entry["round"]
            after_attack = entry["after_attacker_update"]
            after_defense = entry["after_defender_update"]
            effects = entry["update_effects"]
            print(
                f"  {r:>5}  {after_attack['attack_success_rate']:>14.1%}  "
                f"{effects['attacker_success_change']:>+15.4f}  "
                f"{after_defense['auroc']:>15.4f}  "
                f"{effects['defender_accuracy_change']:>+15.4f}"
            )

        print(f"{'=' * 95}")
