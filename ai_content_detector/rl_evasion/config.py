"""Shared configuration for RL evasion experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TextEvasionConfig:
    """Configuration for text-modality RL evasion training."""

    # Generator model — any HF causal LM that supports LoRA fine-tuning.
    # Examples: "Qwen/Qwen3.5-9B-Base", "google/gemma-4-E4B".
    # Must be a local model (RL training requires gradient access).
    generator_model: str = "Qwen/Qwen3.5-9B-Base"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Training
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95

    # GRPO specific
    grpo_num_generations: int = 4  # number of completions per prompt
    grpo_kl_coeff: float = 0.05  # KL penalty coefficient (manual loop only)
    grpo_max_grad_norm: float = 1.0
    grpo_min_output_tokens: int = 20  # reject generations shorter than this

    # Reward weights
    reward_evasion_weight: float = 0.6
    reward_semantic_weight: float = 0.3
    reward_quality_weight: float = 0.1

    # Detector ensemble for reward
    detector_names: List[str] = field(
        default_factory=lambda: ["binoculars", "fast_detect_gpt", "roberta_classifier"]
    )

    # Semantic preservation
    embedding_model: str = "intfloat/e5-base-v2"
    min_semantic_similarity: float = 0.85

    # Reference corpus
    reference_dataset: str = "cnn_dailymail"
    reference_split: str = "train"
    reference_max_samples: int = 10000

    # Output
    output_dir: str = "results/text_evasion"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10

    seed: int = 42


@dataclass
class MultiSPINConfig:
    """Configuration for MultiSPIN distribution matching."""

    # Base SPIN config
    base_model: str = "Qwen/Qwen3.5-9B-Base"
    lora_rank: int = 16
    lora_alpha: int = 32

    # MultiSPIN loss weights
    lambda_spin: float = 1.0
    lambda_stylo: float = 0.5
    lambda_emb: float = 0.3
    lambda_auth: float = 0.2
    lambda_task: float = 0.1

    # Feature matching
    embedding_model: str = "intfloat/e5-base-v2"
    stylometric_features: List[str] = field(
        default_factory=lambda: [
            "burstiness", "ttr", "avg_sent_len", "std_sent_len",
            "fw_ratio", "pos_bigram_entropy", "punct_density", "hapax_ratio",
        ]
    )

    # Training
    num_iterations: int = 5  # SPIN iterations
    steps_per_iteration: int = 1000
    learning_rate: float = 5e-5
    batch_size: int = 4
    max_seq_length: int = 512
    max_new_tokens: int = 200
    temperature: float = 0.8
    gradient_accumulation_steps: int = 4
    beta_spin: float = 0.1  # DPO-style temperature for SPIN loss

    # Reference corpus
    reference_dataset: str = "cnn_dailymail"
    reference_max_samples: int = 5000

    # Style embedding matching (Rivera Soto et al., 2024)
    lambda_style: float = 0.2
    style_embedding_model: str = "rrivera1849/LUAR-MUD"

    # Persistent feature bank
    feature_bank_max_probes: int = 50
    probe_retrain_interval: int = 200

    output_dir: str = "results/multispin"
    seed: int = 42


@dataclass
class ImageEvasionConfig:
    """Configuration for image-modality RL evasion training."""

    # Diffusion model
    diffusion_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    lora_rank: int = 8
    lora_alpha: int = 16

    # DDPO training
    num_train_epochs: int = 50
    train_batch_size: int = 4
    sample_batch_size: int = 8
    num_inference_steps: int = 50
    learning_rate: float = 1e-5
    kl_coeff: float = 0.1

    # DDPO-IS (PPO-style) parameters
    clip_range: float = 1e-4  # PPO clip epsilon
    num_inner_epochs: int = 1  # PPO inner epochs over sampled trajectories
    per_prompt_stat_tracking: bool = True  # per-prompt advantage normalization
    stat_tracking_buffer_size: int = 32  # buffer size for running stats

    # Reward weights
    reward_evasion_weight: float = 0.5
    reward_clip_weight: float = 0.3
    reward_aesthetic_weight: float = 0.2

    # Detector ensemble
    detector_names: List[str] = field(
        default_factory=lambda: ["efficientnet_b4", "clip_classifier", "siglip_detector"]
    )

    # Prompts
    prompt_dataset: str = "Gustavosta/Stable-Diffusion-Prompts"
    num_prompts: int = 200

    output_dir: str = "results/image_evasion"
    seed: int = 42


@dataclass
class ArmsRaceConfig:
    """Configuration for adversarial arms-race experiments."""

    # General
    num_rounds: int = 10
    modality: str = "text"  # "text" or "image"

    # Attacker config
    attacker_steps_per_round: int = 500
    attacker_config: Optional[TextEvasionConfig] = None
    attacker_image_config: Optional[ImageEvasionConfig] = None

    # Defender config (RADAR-style)
    defender_model: str = "roberta-base"
    defender_retrain_samples: int = 2000
    defender_epochs: int = 3
    defender_lr: float = 2e-5

    # Meta-learning (MAML)
    use_meta_learning: bool = False
    meta_lr: float = 1e-4
    meta_inner_steps: int = 5
    meta_outer_steps: int = 100
    meta_second_order: bool = True  # full second-order MAML vs FOMAML
    meta_gradient_clip: float = 1.0

    # Evaluation
    eval_prompts: int = 500
    eval_metrics: List[str] = field(
        default_factory=lambda: ["tpr_at_1fpr", "tpr_at_5fpr", "auroc", "attack_success_rate"]
    )

    output_dir: str = "results/arms_race"
    seed: int = 42
