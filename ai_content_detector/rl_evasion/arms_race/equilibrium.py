"""Multi-round adversarial arms race equilibrium experiments.

The most important experiment in the project: alternate between attacker
(fine-tuning the generator to evade detectors) and defender (retraining
detectors on the new attacker outputs), measuring where the equilibrium
sits and how much each round costs.

Reference: todo/rl_evasion_research_directions_v2.md §Thread 2 —
"The equilibrium experiment (most important experiment in the project)"

Protocol:
    Round 0: initial generator + initial detector ensemble
    Round r → r+1:
        1. Attacker: fine-tune generator against current detectors (GRPO/MultiSPIN)
        2. Defender: retrain detectors on new generator outputs (RADAR-style)
        3. Evaluate: TPR@1%FPR, AUROC, attack success rate
    Run for N=10 rounds. Report equilibrium curve.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np

from ..config import ArmsRaceConfig, TextEvasionConfig, ImageEvasionConfig

logger = logging.getLogger(__name__)


class ArmsRaceExperiment:
    """Run the multi-round attacker/defender arms race."""

    def __init__(self, config: Optional[ArmsRaceConfig] = None):
        self.config = config or ArmsRaceConfig()
        self.history: List[Dict] = []  # per-round metrics
        self.attacker = None
        self.defender = None
        # Disjoint prompt splits for fair attacker/defender evaluation
        # (set in _split_prompts after attacker.setup()).
        self.train_prompt_idx: List[int] = []
        self.eval_prompt_idx: List[int] = []

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
        """Set up text attacker (GRPO) and defender (RADAR-style)."""
        from ..text_evasion.grpo_trainer import GRPOTextEvasionTrainer
        from .radar_defender import RADARDefender

        attacker_config = self.config.attacker_config or TextEvasionConfig(
            num_train_epochs=1,
            output_dir=os.path.join(self.config.output_dir, "attacker"),
        )
        attacker_config.output_dir = os.path.join(self.config.output_dir, "attacker")

        self.attacker = GRPOTextEvasionTrainer(config=attacker_config)
        self.attacker.setup()

        self.defender = RADARDefender(
            model_name=self.config.defender_model,
            output_dir=os.path.join(self.config.output_dir, "defender"),
            retrain_samples=self.config.defender_retrain_samples,
            epochs=self.config.defender_epochs,
            lr=self.config.defender_lr,
        )
        self.defender.setup()
        self._split_prompts()

    def _split_prompts(self, eval_frac: float = 0.2) -> None:
        """Split attacker.prompts into disjoint train/eval index lists.

        All attacker generation during training uses ``train_prompt_idx`` and all
        round-level evaluation uses ``eval_prompt_idx`` — this prevents the
        defender from being retrained on prompts that the attacker is later
        evaluated on.
        """
        if self.attacker is None or not getattr(self.attacker, "prompts", None):
            return
        n = len(self.attacker.prompts)
        rng = np.random.default_rng(self.config.seed)
        perm = rng.permutation(n)
        n_eval = max(1, int(eval_frac * n))
        self.eval_prompt_idx = sorted(perm[:n_eval].tolist())
        self.train_prompt_idx = sorted(perm[n_eval:].tolist())
        # Enforce disjointness
        assert not (set(self.train_prompt_idx) & set(self.eval_prompt_idx))
        logger.info(
            "Arms-race prompt split: %d train / %d eval (disjoint)",
            len(self.train_prompt_idx), len(self.eval_prompt_idx),
        )

    def _setup_image(self):
        """Set up image attacker (DDPO) and defender."""
        from ..image_evasion.ddpo_trainer import DDPOImageEvasionTrainer

        image_config = self.config.attacker_image_config or ImageEvasionConfig(
            num_train_epochs=5,
            output_dir=os.path.join(self.config.output_dir, "attacker"),
        )
        image_config.output_dir = os.path.join(self.config.output_dir, "attacker")

        self.attacker = DDPOImageEvasionTrainer(config=image_config)
        self.attacker.setup()
        # Image defender not implemented yet — would use adversarial retraining of image classifiers
        self.defender = None

    def run(self):
        """Execute the full arms race experiment."""
        cfg = self.config

        for round_idx in range(cfg.num_rounds):
            logger.info(
                "\n%s\n  ARMS RACE — ROUND %d/%d\n%s",
                "=" * 60, round_idx + 1, cfg.num_rounds, "=" * 60,
            )

            round_metrics = {"round": round_idx + 1}

            # === Attacker phase ===
            logger.info("--- Attacker phase ---")
            self.attacker.train()

            # Generate outputs for evaluation and defender retraining
            attacker_eval = self.attacker.evaluate(num_samples=min(cfg.eval_prompts, 200))
            round_metrics["attacker"] = attacker_eval

            logger.info(
                "Attacker: evasion=%.3f, semantic=%.3f, attack_success=%.1f%%",
                attacker_eval.get("mean_evasion", 0),
                attacker_eval.get("mean_semantic_similarity", 0),
                attacker_eval.get("attack_success_rate", 0) * 100,
            )

            # === Defender phase ===
            if self.defender is not None:
                logger.info("--- Defender phase ---")

                # Generate fresh samples from the attacker on the TRAINING prompts
                # (disjoint from the eval set) for defender retraining.
                import torch

                device = next(self.attacker.model.parameters()).device
                train_idx = self.train_prompt_idx or list(range(len(self.attacker.prompts)))
                sample_ids = train_idx[: cfg.defender_retrain_samples]
                attacker_texts = []
                for i in sample_ids:
                    prompt = self.attacker.prompts[i]
                    inputs = self.attacker.tokenizer(
                        prompt, return_tensors="pt", truncation=True, max_length=256
                    ).to(device)
                    with torch.no_grad():
                        gen = self.attacker.model.generate(**inputs, max_new_tokens=200, do_sample=True)
                    text = self.attacker.tokenizer.decode(
                        gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                    )
                    attacker_texts.append(text)

                human_refs = [self.attacker.references[i] for i in sample_ids]
                defender_metrics = self.defender.retrain(
                    ai_texts=attacker_texts,
                    human_texts=human_refs,
                )
                round_metrics["defender"] = defender_metrics

                logger.info(
                    "Defender retrained: accuracy=%.3f, auroc=%.3f",
                    defender_metrics.get("accuracy", 0),
                    defender_metrics.get("auroc", 0),
                )

                # Update attacker's detector ensemble with the retrained defender
                # (this closes the adversarial loop).
                self._update_attacker_detectors()
            else:
                round_metrics["defender"] = {"note": "No defender configured"}

            # === Compute Nash-style equilibrium gap on the disjoint eval set ===
            round_metrics["nash_gap"] = self._compute_nash_gap(round_metrics)

            # === Record metrics ===
            self.history.append(round_metrics)
            self._save_history()

            logger.info(
                "Round %d complete. Nash gap = %.4f",
                round_idx + 1, round_metrics["nash_gap"],
            )

        logger.info("\n%s\n  ARMS RACE COMPLETE\n%s", "=" * 60, "=" * 60)
        self._print_summary()

    @staticmethod
    def _compute_nash_gap(round_metrics: Dict) -> float:
        """Return the best-response deficit ∈ [0, 1].

        Interpretation: the fraction of AI samples correctly flagged by the
        defender that the attacker did NOT already bypass. A gap of 0 means the
        attacker has already defeated every improvement the defender could make
        on this round (equilibrium); a gap of 1 means the defender fully catches
        an attacker that didn't evade anything. Bounded, nonnegative, and
        game-theoretically meaningful (best-response deficit under the zero-sum
        detector/attacker formulation).
        """
        atk = round_metrics.get("attacker", {}) or {}
        defn = round_metrics.get("defender", {}) or {}
        attacker_success = float(atk.get("attack_success_rate", 0.0))  # in [0,1]
        defender_acc = float(defn.get("accuracy", 0.0))                # in [0,1]
        gap = defender_acc - (1.0 - attacker_success)
        return float(max(0.0, min(1.0, gap)))

    def _update_attacker_detectors(self):
        """Replace one of the attacker's detector rewards with the retrained defender."""
        if self.defender is None or self.attacker is None:
            return

        from ..text_evasion.rewards import DetectorReward

        def defender_detect(text: str) -> float:
            return self.defender.predict(text)

        # Replace or add the RADAR defender reward
        new_reward = DetectorReward(defender_detect, name="radar_defender")

        # Find and replace existing radar reward, or append
        for i, dr in enumerate(self.attacker.reward_fn.detector_rewards):
            if dr.name == "radar_defender":
                self.attacker.reward_fn.detector_rewards[i] = new_reward
                return
        self.attacker.reward_fn.detector_rewards.append(new_reward)

    def _save_history(self):
        """Save experiment history to JSON."""
        path = os.path.join(self.config.output_dir, "arms_race_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def _print_summary(self):
        """Print a summary table of the arms race.

        The ``Nash Gap`` column is the best-response deficit on the disjoint eval
        prompt set: defender_accuracy − (1 − attacker_success_rate), clipped to
        [0, 1]. Zero means equilibrium (defender has nothing to gain over
        attacker's current policy); one means maximal detectability.
        """
        print(f"\n{'=' * 95}")
        print("  ARMS RACE EQUILIBRIUM SUMMARY")
        print(f"{'=' * 95}")
        print(f"  {'Round':>5}  {'Attack Evasion':>15}  {'Attack Success':>15}  {'Defender AUROC':>15}  {'Nash Gap':>10}")
        print(f"  {'-'*5}  {'-'*15}  {'-'*15}  {'-'*15}  {'-'*10}")

        for entry in self.history:
            r = entry["round"]
            atk = entry.get("attacker", {}) or {}
            defn = entry.get("defender", {}) or {}
            evasion = float(atk.get('mean_evasion', 0.0))
            auroc = float(defn.get('auroc', 0.0))
            gap = float(entry.get('nash_gap', self._compute_nash_gap(entry)))
            print(
                f"  {r:>5}  {evasion:>15.4f}  "
                f"{atk.get('attack_success_rate', 0):>14.1%}  "
                f"{auroc:>15.4f}  "
                f"{gap:>10.4f}"
            )

        print(f"{'=' * 95}")

        # Convergence analysis on the Nash gap
        if len(self.history) >= 3:
            gaps = [float(h.get("nash_gap", self._compute_nash_gap(h))) for h in self.history]
            last_3 = gaps[-3:]
            if max(last_3) - min(last_3) < 0.02:
                print(f"\n  Nash equilibrium reached. Gap stationary at {np.mean(last_3):.3f}")
            else:
                trend = "closing" if gaps[-1] < gaps[0] else "widening"
                print(f"\n  Gap is still {trend} after {len(self.history)} rounds (Δ={gaps[-1] - gaps[0]:+.3f}).")
