"""CLI entry point for all RL evasion experiments.

Usage:
    # Text evasion with GRPO
    python -m ai_content_detector.rl_evasion.run_experiments --experiment grpo_text

    # SPIN with feature monitoring
    python -m ai_content_detector.rl_evasion.run_experiments --experiment multispin

    # Image evasion with DDPO
    python -m ai_content_detector.rl_evasion.run_experiments --experiment ddpo_image

    # Arms race (text)
    python -m ai_content_detector.rl_evasion.run_experiments --experiment arms_race --modality text

    # Arms race (image)
    python -m ai_content_detector.rl_evasion.run_experiments --experiment arms_race --modality image

    # Smoke test (fast, minimal)
    python -m ai_content_detector.rl_evasion.run_experiments --experiment smoke_test
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_grpo_text(args):
    """Run GRPO text evasion training."""
    from .config import TextEvasionConfig
    from .text_evasion.grpo_trainer import GRPOTextEvasionTrainer

    config = TextEvasionConfig(
        output_dir=args.output_dir or "results/text_evasion/grpo",
        num_train_epochs=args.epochs or 3,
        generator_model=args.model or "Qwen/Qwen3.5-9B-Base",
    )

    trainer = GRPOTextEvasionTrainer(config)
    trainer.setup()
    trainer.train()

    metrics = trainer.evaluate()
    from .text_evasion.evaluate import print_evaluation_report
    print_evaluation_report(metrics, "GRPO Text Evasion")


def run_multispin(args):
    """Run SPIN with feature monitoring."""
    from .config import MultiSPINConfig
    from .text_evasion.multispin import MultiSPINTrainer

    config = MultiSPINConfig(
        output_dir=args.output_dir or "results/text_evasion/multispin",
        num_iterations=args.epochs or 5,
        base_model=args.model or "Qwen/Qwen3.5-9B-Base",
    )

    trainer = MultiSPINTrainer(config)
    trainer.setup()
    trainer.train()


def run_ddpo_image(args):
    """Run DDPO image evasion training."""
    from .config import ImageEvasionConfig
    from .image_evasion.ddpo_trainer import DDPOImageEvasionTrainer

    config = ImageEvasionConfig(
        output_dir=args.output_dir or "results/image_evasion/ddpo",
        num_train_epochs=args.epochs or 50,
    )

    trainer = DDPOImageEvasionTrainer(config)
    trainer.setup()
    trainer.train()

    metrics = trainer.evaluate()
    from .image_evasion.evaluate import print_image_evaluation_report
    print_image_evaluation_report(metrics, "DDPO Image Evasion")


def run_arms_race(args):
    """Run the adversarial arms race experiment."""
    from .config import ArmsRaceConfig
    from .arms_race.equilibrium import ArmsRaceExperiment

    config = ArmsRaceConfig(
        num_rounds=args.rounds or 10,
        modality=args.modality or "text",
        output_dir=args.output_dir or f"results/arms_race/{args.modality or 'text'}",
    )

    experiment = ArmsRaceExperiment(config)
    experiment.setup()
    experiment.run()


def run_meta_adapt(args):
    """Run MAML meta-learning adaptation experiment."""
    from .config import ArmsRaceConfig, TextEvasionConfig
    from .arms_race.meta_adapt import MAMLAdaptation, DetectorZoo

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset

    model_name = args.model or "Qwen/Qwen3.5-9B-Base"
    output_dir = args.output_dir or "results/meta_adapt"
    outer_steps = args.epochs or 100

    logger.info("Loading model %s for meta-learning...", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )

    # LoRA: try "all-linear" first, fall back for hybrid architectures
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    try:
        model = get_peft_model(model, lora_config)
    except (ValueError, RuntimeError):
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Build detector zoo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zoo = DetectorZoo.build_default(device=device)
    logger.info("Detector zoo has %d detectors", len(zoo.detectors))
    if len(zoo.detectors) < 2:
        raise RuntimeError(
            "Meta-adaptation requires at least two functioning detectors so one can be held out"
        )
    held_out_name = sorted(zoo.detectors)[-1]
    held_out_fn = zoo.detectors.pop(held_out_name)
    logger.info("Reserved detector before meta-training: %s", held_out_name)

    # Load prompts
    ds = load_dataset("cnn_dailymail", "3.0.0", split="train")
    prompts = [" ".join(t.split()[:50]) for t in ds["article"][:500] if len(t.split()) > 30]

    # Run meta-learning
    maml = MAMLAdaptation(
        model=model,
        tokenizer=tokenizer,
        detector_zoo=zoo,
        meta_lr=1e-4,
        inner_lr=5e-5,
        inner_steps=5,
        outer_steps=outer_steps,
        detectors_per_episode=min(3, len(zoo.detectors)),
        first_order=args.first_order,
        output_dir=output_dir,
    )
    maml.train(prompts)

    logger.info("Measuring adaptation cost to held-out detector: %s", held_out_name)
    comparison = maml.compare_with_pre_meta(held_out_fn, held_out_name, prompts[:100])
    logger.info("Speedup: %.2fx", comparison["speedup"])


def run_benchmark(args):
    """Run evasion benchmark: RL methods vs baselines."""
    from .benchmarking.benchmark import BenchmarkRunner, GRPOEvasionMethod, MultiSPINEvasionMethod
    from .benchmarking.datasets import load_dataset_by_name
    from .benchmarking.baselines import get_all_baselines, get_lightweight_baselines

    dataset = load_dataset_by_name(args.dataset, max_samples=args.epochs or 200)

    model_name = args.model or "Qwen/Qwen3.5-9B-Base"

    import torch
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        methods = get_all_baselines(device="auto", model_name=model_name)
    else:
        methods = get_lightweight_baselines()
        logger.info("No GPU detected — using lightweight baselines only")

    # Add RL methods if checkpoints exist
    grpo_dir = os.path.join(args.output_dir or "results", "text_evasion", "grpo", "final")
    multispin_dir = os.path.join(args.output_dir or "results", "text_evasion", "multispin", "iteration_5")

    if os.path.isdir(grpo_dir):
        methods.append(GRPOEvasionMethod(grpo_dir))
        logger.info("Found GRPO checkpoint at %s", grpo_dir)
    else:
        logger.info("No GRPO checkpoint found at %s — skipping", grpo_dir)

    if os.path.isdir(multispin_dir):
        methods.append(MultiSPINEvasionMethod(multispin_dir))
        logger.info("Found MultiSPIN checkpoint at %s", multispin_dir)
    else:
        logger.info("No MultiSPIN checkpoint found at %s — skipping", multispin_dir)

    runner = BenchmarkRunner(
        methods=methods,
        dataset=dataset,
        device="auto",
    )
    runner.setup_detectors()
    runner.run()
    runner.print_comparison()
    runner.save_results(args.output_dir or "results/benchmark")


def run_smoke_test(args):
    """Run a minimal smoke test to verify all components load."""
    print("=== Smoke Test: AI Content Detector ===\n")

    # 1. Test detector imports
    print("1. Testing detector imports...")
    try:
        from ..detectors.ensemble import BaseDetector, DetectionResult, EnsembleAggregator
        print("   Ensemble imports: OK")
    except Exception as e:
        print(f"   Ensemble imports: FAILED ({e})")

    try:
        from ..detectors.text_detectors import DivEyeDetector
        DivEyeDetector(scoring_model="gpt2", device="cpu")
        print("   DivEye constructor: OK (inference still requires a fitted classifier)")
    except Exception as e:
        print(f"   DivEye detector: FAILED ({e})")

    # 2. Test config
    print("\n2. Testing configs...")
    try:
        from .config import TextEvasionConfig, MultiSPINConfig, ImageEvasionConfig, ArmsRaceConfig
        cfg = TextEvasionConfig()
        print(f"   TextEvasionConfig: OK (model={cfg.generator_model})")
    except Exception as e:
        print(f"   Config: FAILED ({e})")

    # 3. Test reward functions
    print("\n3. Testing reward functions...")
    try:
        from .text_evasion.rewards import DetectorReward, CompositeReward
        dummy = DetectorReward(lambda t: 0.5, name="dummy")
        r = dummy("test text")
        print(f"   DetectorReward: OK (reward={r})")
    except Exception as e:
        print(f"   Reward: FAILED ({e})")

    # 4. Test feature bank
    print("\n4. Testing feature bank...")
    try:
        from .text_evasion.feature_bank import FeatureBank
        bank = FeatureBank(max_probes=10)
        print(f"   FeatureBank: OK ({bank.summary()})")
    except Exception as e:
        print(f"   FeatureBank: FAILED ({e})")

    # 5. Test adaptive classifier defender
    print("\n5. Testing adaptive classifier defender...")
    try:
        from .arms_race.radar_defender import AdaptiveClassifierDefender
        print("   AdaptiveClassifierDefender import: OK")
    except Exception as e:
        print(f"   AdaptiveClassifierDefender: FAILED ({e})")

    # 6. Test style embedding detector
    print("\n6. Testing style embedding detector...")
    try:
        from ..detectors.style_detector import StyleEmbeddingDetector
        det = StyleEmbeddingDetector(device="cpu")
        print(f"   StyleEmbeddingDetector import: OK (model={det._model_name})")
    except Exception as e:
        print(f"   StyleEmbeddingDetector: FAILED ({e})")

    # 7. Test MAML meta-adaptation
    print("\n7. Testing MAML meta-adaptation...")
    try:
        from .arms_race.meta_adapt import MAMLAdaptation, DetectorZoo
        zoo = DetectorZoo()
        print(f"   MAMLAdaptation import: OK")
    except Exception as e:
        print(f"   MAMLAdaptation: FAILED ({e})")

    # 8. Test DDPO trainer (full version)
    print("\n8. Testing DDPO trainer (full trajectory tracking)...")
    try:
        from .image_evasion.ddpo_trainer import (
            DDPOImageEvasionTrainer,
            ddim_step_with_logprob,
            PerPromptStatTracker,
        )
        tracker = PerPromptStatTracker(buffer_size=16)
        print(f"   DDPO full trainer import: OK")
    except Exception as e:
        print(f"   DDPO trainer: FAILED ({e})")

    print("\n=== Smoke Test Complete ===")


def main():
    parser = argparse.ArgumentParser(
        description="RL Evasion Experiments for AI Content Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["grpo_text", "multispin", "ddpo_image", "arms_race", "meta_adapt", "benchmark", "smoke_test"],
        help="Which experiment to run",
    )
    parser.add_argument("--model", type=str, default=None, help="Base model name")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--rounds", type=int, default=None, help="Arms race rounds")
    parser.add_argument("--modality", type=str, default="text", choices=["text", "image"])
    parser.add_argument(
        "--first-order", action="store_true",
        help="meta_adapt: use FOMAML (first-order) instead of full second-order MAML",
    )
    parser.add_argument(
        "--dataset", type=str, default="hc3", choices=["hc3", "cnn_dailymail"],
        help="benchmark: which dataset to evaluate on",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    dispatch = {
        "grpo_text": run_grpo_text,
        "multispin": run_multispin,
        "ddpo_image": run_ddpo_image,
        "arms_race": run_arms_race,
        "meta_adapt": run_meta_adapt,
        "benchmark": run_benchmark,
        "smoke_test": run_smoke_test,
    }

    dispatch[args.experiment](args)


if __name__ == "__main__":
    main()
