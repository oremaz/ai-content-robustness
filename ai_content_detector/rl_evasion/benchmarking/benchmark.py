"""Benchmark runner: compare RL evasion methods vs baselines.

Evaluates each method on a dataset, measures evasion success against
the detector ensemble, semantic preservation, and quality, then
produces a comparison table.

Usage:
    python -m ai_content_detector.rl_evasion.benchmarking.benchmark \
        --dataset hc3 --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..text_evasion.evaluate import evaluate_text_evasion
from ..text_evasion.rewards import CompositeReward, build_detector_reward_from_name
from .baselines import (
    BaseEvasionMethod,
    EvasionCapability,
    get_all_baselines,
    get_lightweight_baselines,
)
from .datasets import BenchmarkDataset, load_dataset_by_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RL method wrappers (so they share the BaseEvasionMethod interface)
# ---------------------------------------------------------------------------


class GRPOEvasionMethod(BaseEvasionMethod):
    """Wrapper around a trained GRPO model for benchmarking."""

    name = "grpo_rl"
    capabilities = frozenset({EvasionCapability.GENERATE})

    def __init__(self, checkpoint_dir: str, device: str = "auto", max_new_tokens: int = 256):
        self._checkpoint_dir = checkpoint_dir
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self.optimized_detector_names = self._load_training_detectors()

    def _load_training_detectors(self) -> frozenset[str]:
        metadata_path = os.path.join(self._checkpoint_dir, "training_metadata.json")
        if not os.path.exists(metadata_path):
            return frozenset()
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)
        return frozenset(metadata.get("detector_names", []))

    def provenance(self) -> dict:
        return {**super().provenance(), "checkpoint_dir": self._checkpoint_dir}

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint_dir, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        # Try loading as a PEFT model, fall back to full model
        try:
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(self._checkpoint_dir)
            base = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
            )
            self._model = PeftModel.from_pretrained(base, self._checkpoint_dir)
        except Exception:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._checkpoint_dir, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
            )
        self._model.eval()
        logger.info("GRPO model loaded from %s", self._checkpoint_dir)

    def generate(self, prompt: str) -> str:
        self._load()
        import torch

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(next(self._model.parameters()).device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens,
                do_sample=True, temperature=0.8, top_p=0.95,
            )
        return self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class MultiSPINEvasionMethod(BaseEvasionMethod):
    """Wrapper around a trained MultiSPIN model for benchmarking."""

    name = "multispin"
    capabilities = frozenset({EvasionCapability.GENERATE})

    def __init__(self, checkpoint_dir: str, device: str = "auto", max_new_tokens: int = 256):
        self._checkpoint_dir = checkpoint_dir
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self.optimized_detector_names = self._load_training_detectors()

    def _load_training_detectors(self) -> frozenset[str]:
        metadata_path = os.path.join(self._checkpoint_dir, "training_metadata.json")
        if not os.path.exists(metadata_path):
            return frozenset()
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)
        return frozenset(metadata.get("detector_names", []))

    def provenance(self) -> dict:
        return {**super().provenance(), "checkpoint_dir": self._checkpoint_dir}

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint_dir, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(self._checkpoint_dir)
            base = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
            )
            self._model = PeftModel.from_pretrained(base, self._checkpoint_dir)
        except Exception:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._checkpoint_dir, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
            )
        self._model.eval()
        logger.info("MultiSPIN model loaded from %s", self._checkpoint_dir)

    def generate(self, prompt: str) -> str:
        self._load()
        import torch

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(next(self._model.parameters()).device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens,
                do_sample=True, temperature=0.8, top_p=0.95,
            )
        return self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class MethodApplication:
    status: str
    track: str
    generated_texts: List[str] = field(default_factory=list)
    source_texts: List[str] = field(default_factory=list)
    sample_indices: List[int] = field(default_factory=list)
    errors: List[Dict[str, object]] = field(default_factory=list)
    attempted: int = 0


class BenchmarkRunner:
    """Run evasion benchmarks across methods and datasets.

    Detector isolation: by convention we split ``detector_names`` into two
    disjoint sets — ``train_detectors`` (those a method may have been trained
    against) and ``heldout_detectors`` (used only for evaluation). The final
    evasion metric is computed on both, but a separate "held-out evasion" key
    is surfaced so the main comparison table does not over-credit methods that
    already optimized against the eval detector pool.
    """

    def __init__(
        self,
        methods: List[BaseEvasionMethod],
        dataset: BenchmarkDataset,
        detector_names: Optional[List[str]] = None,
        device: str = "auto",
        heldout_detectors: Optional[List[str]] = None,
    ):
        self.methods = methods
        self.dataset = dataset
        self.detector_names = detector_names or ["binoculars", "fast_detect_gpt", "diveye"]
        self.heldout_detectors = list(heldout_detectors or [])
        overlap = set(self.heldout_detectors) & set(self.detector_names)
        if overlap:
            raise ValueError(
                f"heldout_detectors must be disjoint from detector_names; overlap={sorted(overlap)}"
            )
        self.device = device
        self.reward_fn = None
        self.heldout_reward_fn = None
        self.results: Dict[str, Dict] = {}

    def setup_detectors(self):
        """Build both the evaluation ensemble and (optionally) the held-out ensemble."""
        from ..text_evasion.rewards import SemanticSimilarityReward

        def _build_ensemble(names: List[str], label: str) -> CompositeReward:
            rewards = []
            for name in names:
                try:
                    dr = build_detector_reward_from_name(name, device=self.device)
                    rewards.append(dr)
                    logger.info("Loaded %s detector: %s", label, name)
                except Exception as e:
                    logger.warning("Could not load %s detector %s: %s", label, name, e)
            if not rewards:
                raise RuntimeError(
                    f"No {label} detector could be loaded from requested names: {names}"
                )
            return CompositeReward(
                detector_rewards=rewards,
                semantic_reward=SemanticSimilarityReward(device=self.device),
            )

        self.reward_fn = _build_ensemble(self.detector_names, "eval")
        if self.heldout_detectors:
            self.heldout_reward_fn = _build_ensemble(self.heldout_detectors, "held-out")
        else:
            self.heldout_reward_fn = None

    def run(self) -> Dict[str, Dict]:
        """Run all methods and collect results."""
        if self.reward_fn is None:
            self.setup_detectors()

        for method in self.methods:
            logger.info("=== Evaluating: %s ===", method.name)
            t0 = time.time()
            overlap = set(method.optimized_detector_names) & set(self.heldout_detectors)
            if overlap:
                raise ValueError(
                    f"Method {method.name} was optimized against held-out detectors: {sorted(overlap)}"
                )

            application = self._apply_method(method)
            if application.status == "skipped":
                self.results[method.name] = {
                    "status": "skipped",
                    "track": application.track,
                    "reason": application.errors[0]["error"],
                    "num_attempted": 0,
                    "num_succeeded": 0,
                    "num_failed": 0,
                }
                logger.warning("Skipping %s: %s", method.name, application.errors[0]["error"])
                continue
            if not application.generated_texts:
                self.results[method.name] = {
                    "status": "failed",
                    "track": application.track,
                    "num_attempted": application.attempted,
                    "num_succeeded": 0,
                    "num_failed": len(application.errors),
                    "errors": application.errors,
                    "wall_time_sec": time.time() - t0,
                }
                logger.error("Method %s failed on every attempted sample", method.name)
                continue

            metrics = evaluate_text_evasion(
                generated_texts=application.generated_texts,
                source_texts=application.source_texts,
                reward_fn=self.reward_fn,
            )
            if self.heldout_reward_fn is not None:
                heldout_metrics = evaluate_text_evasion(
                    generated_texts=application.generated_texts,
                    source_texts=application.source_texts,
                    reward_fn=self.heldout_reward_fn,
                )
                metrics["heldout_mean_evasion"] = heldout_metrics["mean_evasion"]
                metrics["heldout_attack_success_rate"] = heldout_metrics["attack_success_rate"]
                for k, v in heldout_metrics.items():
                    if k.startswith("evasion_"):
                        metrics[f"heldout_{k}"] = v
            metrics["wall_time_sec"] = time.time() - t0
            metrics.update({
                "status": "completed" if not application.errors else "partial",
                "track": application.track,
                "num_attempted": application.attempted,
                "num_succeeded": len(application.generated_texts),
                "num_failed": len(application.errors),
                "sample_indices": application.sample_indices,
                "errors": application.errors,
            })
            self.results[method.name] = metrics

            logger.info(
                "  %s: evasion=%.3f, semantic=%.3f, attack_success=%.1f%%",
                method.name,
                metrics["mean_evasion"],
                metrics["mean_semantic_similarity"],
                metrics["attack_success_rate"] * 100,
            )

        return self.results

    def _apply_method(self, method: BaseEvasionMethod) -> MethodApplication:
        """Apply an evasion method to the dataset.

        Contract:
        - Datasets where ``ai_texts_available`` is True provide genuine AI-generated
          samples that post-hoc methods (paraphrase, synonym sub) can operate on.
        - Datasets where it's False (e.g., CNN/DailyMail prompts) only expose
          prompts; methods that only implement ``evade`` are skipped and a warning
          is logged instead of being fed empty strings.
        """
        ai_available = getattr(self.dataset, "ai_texts_available", True)
        capability = EvasionCapability.REWRITE if ai_available else EvasionCapability.GENERATE
        track = capability.value
        if capability not in method.capabilities:
            return MethodApplication(
                status="skipped",
                track=track,
                errors=[{
                    "sample_index": None,
                    "error": f"method does not support {track}",
                }],
            )

        application = MethodApplication(status="completed", track=track)
        for i, (prompt, ai_text) in enumerate(zip(self.dataset.prompts, self.dataset.ai_texts)):
            application.attempted += 1
            try:
                if ai_available:
                    evaded = method.evade(ai_text)
                    source = ai_text
                else:
                    evaded = method.generate(prompt)
                    source = (
                        self.dataset.human_references[i]
                        if self.dataset.human_references else prompt
                    )
                if not isinstance(evaded, str) or not evaded.strip():
                    raise ValueError("method returned an empty or non-string output")
                application.generated_texts.append(evaded)
                application.source_texts.append(source)
                application.sample_indices.append(i)
            except Exception as e:
                logger.warning("Method %s failed on sample %d: %s", method.name, i, e)
                application.errors.append({
                    "sample_index": i,
                    "error_type": type(e).__name__,
                    "error": str(e),
                })

        if application.errors:
            application.status = "partial" if application.generated_texts else "failed"
        return application

    def print_comparison(self):
        """Print a formatted comparison table."""
        if not self.results:
            print("No results to display.")
            return

        print(f"\n{'=' * 90}")
        print(f"  EVASION BENCHMARK — {self.dataset.name} ({len(self.dataset)} samples)")
        print(f"{'=' * 90}")
        print(
            f"  {'Method':<20s}  {'Evasion':>8s}  {'Semantic':>9s}  "
            f"{'Attack SR':>10s}  {'Det. Score':>10s}  {'Time (s)':>9s}"
        )
        print(f"  {'-'*20}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*9}")

        for name, m in self.results.items():
            if m.get("status") in {"skipped", "failed"}:
                print(f"  {name:<20s}  {m['status']:>8s}  {m.get('reason', '')}")
                continue
            print(
                f"  {name:<20s}  {m['mean_evasion']:>8.4f}  "
                f"{m['mean_semantic_similarity']:>9.4f}  "
                f"{m['attack_success_rate']:>9.1%}  "
                f"{m.get('mean_detector_ai_score', 0):>10.4f}  "
                f"{m.get('wall_time_sec', 0):>9.1f}"
            )

        print(f"{'=' * 90}")

        # Per-detector breakdown
        det_keys = set()
        for m in self.results.values():
            det_keys.update(k for k in m if k.startswith("evasion_"))

        if det_keys:
            print(f"\n  Per-detector evasion scores:")
            header = f"  {'Method':<20s}" + "".join(f"  {k.replace('evasion_', ''):>15s}" for k in sorted(det_keys))
            print(header)
            for name, m in self.results.items():
                row = f"  {name:<20s}"
                for k in sorted(det_keys):
                    row += f"  {m.get(k, 0):>15.4f}"
                print(row)
            print()

    def save_results(self, output_dir: str):
        """Save results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "benchmark_results.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "dataset": self.dataset.name,
                    "dataset_revision": self.dataset.source_revision,
                    "sample_ids": self.dataset.sample_ids,
                    "num_samples": len(self.dataset),
                    "detectors": self.detector_names,
                    "heldout_detectors": self.heldout_detectors,
                    "method_provenance": {
                        method.name: method.provenance() for method in self.methods
                    },
                    "results": self.results,
                },
                f,
                indent=2,
                default=str,
            )
        logger.info("Results saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evasion benchmark: RL methods vs baselines")
    parser.add_argument("--dataset", type=str, default="hc3", choices=["hc3", "cnn_dailymail"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/benchmark")
    parser.add_argument(
        "--lightweight", action="store_true",
        help="Use only lightweight baselines (no GPU, no large models)",
    )
    parser.add_argument("--grpo-checkpoint", type=str, default=None, help="Path to trained GRPO checkpoint")
    parser.add_argument("--multispin-checkpoint", type=str, default=None, help="Path to trained MultiSPIN checkpoint")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B-Base", help="Base model for generation baselines")
    parser.add_argument(
        "--detectors", type=str, nargs="+",
        default=["binoculars", "fast_detect_gpt", "diveye"],
        help="Detector names for evaluation (attackers may have trained against these)",
    )
    parser.add_argument(
        "--heldout-detectors", type=str, nargs="+", default=[],
        help=(
            "Detectors reserved strictly for held-out evaluation. Must be disjoint "
            "from --detectors. Exposed as heldout_* keys in the results."
        ),
    )

    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset_by_name(args.dataset, max_samples=args.max_samples)

    # Build method list
    if args.lightweight:
        methods = get_lightweight_baselines()
    else:
        methods = get_all_baselines(device=args.device, model_name=args.model)

    # Add RL methods if checkpoints provided
    if args.grpo_checkpoint:
        methods.append(GRPOEvasionMethod(args.grpo_checkpoint, device=args.device))
    if args.multispin_checkpoint:
        methods.append(MultiSPINEvasionMethod(args.multispin_checkpoint, device=args.device))

    # Run benchmark
    runner = BenchmarkRunner(
        methods=methods,
        dataset=dataset,
        detector_names=args.detectors,
        heldout_detectors=args.heldout_detectors,
        device=args.device,
    )
    runner.setup_detectors()
    runner.run()
    runner.print_comparison()
    runner.save_results(args.output_dir)


if __name__ == "__main__":
    main()
