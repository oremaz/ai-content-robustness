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
import logging
import os
from typing import Dict, List, Optional

import numpy as np

from ..config import ArmsRaceConfig, TextEvasionConfig

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

    def _partition_prompts(self, eval_frac: float = 0.2) -> None:
        """Create an immutable split and remove evaluation prompts from training."""
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

    def _generate(self, prompts: List[str], seed: int) -> List[str]:
        import torch

        device = next(self.attacker.model.parameters()).device
        self.attacker.model.eval()
        generator = torch.Generator(device=device).manual_seed(seed)
        texts = []
        for prompt in prompts:
            inputs = self.attacker.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.attacker.config.max_seq_length,
            ).to(device)
            with torch.no_grad():
                generated = self.attacker.model.generate(
                    **inputs,
                    max_new_tokens=self.attacker.config.max_new_tokens,
                    temperature=self.attacker.config.temperature,
                    top_p=self.attacker.config.top_p,
                    do_sample=True,
                    generator=generator,
                )
            texts.append(self.attacker.tokenizer.decode(
                generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            ))
        return texts

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
        """Execute the full arms race experiment."""
        cfg = self.config

        for round_idx in range(cfg.num_rounds):
            logger.info(
                "\n%s\n  ARMS RACE — ROUND %d/%d\n%s",
                "=" * 60, round_idx + 1, cfg.num_rounds, "=" * 60,
            )

            round_metrics = {"round": round_idx + 1, "eval_samples": len(self.eval_prompts)}

            eval_prompts = self.eval_prompts[:min(cfg.eval_prompts, len(self.eval_prompts))]
            pre_attack_texts = self._generate(
                eval_prompts, seed=cfg.seed + 10_000 * round_idx,
            )
            before_attack = self._evaluate_defender(pre_attack_texts)

            # === Attacker phase ===
            logger.info("--- Attacker phase ---")
            self.attacker.train(max_steps=cfg.attacker_steps_per_round)
            post_attack_texts = self._generate(
                eval_prompts, seed=cfg.seed + 10_000 * round_idx,
            )
            after_attack = self._evaluate_defender(post_attack_texts)
            round_metrics["before_attacker_update"] = before_attack
            round_metrics["after_attacker_update"] = after_attack

            logger.info(
                "Attacker update: held-out attack success=%.1f%%, detector accuracy=%.3f",
                after_attack.get("attack_success_rate", 0) * 100,
                after_attack.get("accuracy", 0),
            )

            # === Defender phase ===
            if self.defender is not None:
                logger.info("--- Defender phase ---")

                n_train = min(cfg.defender_retrain_samples // 2, len(self.train_prompts))
                attacker_texts = self._generate(
                    self.train_prompts[:n_train], seed=cfg.seed + 10_000 * round_idx + 2,
                )
                human_refs = self.train_references[:n_train]
                defender_metrics = self.defender.retrain(
                    ai_texts=attacker_texts,
                    human_texts=human_refs,
                )
                round_metrics["defender_training"] = defender_metrics

                logger.info(
                    "Defender retrained: accuracy=%.3f, auroc=%.3f",
                    defender_metrics.get("accuracy", 0),
                    defender_metrics.get("auroc", 0),
                )

                # Update attacker's detector ensemble with the retrained defender
                # (this closes the adversarial loop).
                self._update_attacker_detectors()
            after_defense = self._evaluate_defender(post_attack_texts)
            round_metrics["after_defender_update"] = after_defense
            round_metrics["update_effects"] = self._compute_update_effects(
                before_attack, after_attack, after_defense,
            )

            # === Record metrics ===
            self.history.append(round_metrics)
            self._save_history()

            logger.info(
                "Round %d complete. Attacker change = %+.4f, defender accuracy change = %+.4f",
                round_idx + 1,
                round_metrics["update_effects"]["attacker_success_change"],
                round_metrics["update_effects"]["defender_accuracy_change"],
            )

        logger.info("\n%s\n  ARMS RACE COMPLETE\n%s", "=" * 60, "=" * 60)
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
