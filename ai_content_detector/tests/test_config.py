"""Tests for configuration dataclasses."""

from __future__ import annotations

import pytest

from ai_content_detector.rl_evasion.config import (
    ArmsRaceConfig,
    ImageEvasionConfig,
    MultiSPINConfig,
    TextEvasionConfig,
)


class TestTextEvasionConfig:
    def test_defaults(self):
        cfg = TextEvasionConfig()
        assert cfg.generator_model == "Qwen/Qwen3.5-9B-Base"
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == 32
        assert cfg.reward_evasion_weight == 0.6
        assert cfg.reward_semantic_weight == 0.3
        assert cfg.reward_quality_weight == 0.1
        assert cfg.seed == 42

    def test_override(self):
        cfg = TextEvasionConfig(generator_model="gpt2", lora_rank=8, seed=123)
        assert cfg.generator_model == "gpt2"
        assert cfg.lora_rank == 8
        assert cfg.seed == 123

    def test_reward_weights_sum_to_one(self):
        cfg = TextEvasionConfig()
        total = cfg.reward_evasion_weight + cfg.reward_semantic_weight + cfg.reward_quality_weight
        assert total == pytest.approx(1.0)

    def test_detector_names_list(self):
        cfg = TextEvasionConfig()
        assert isinstance(cfg.detector_names, list)
        assert len(cfg.detector_names) == 3
        assert "binoculars" in cfg.detector_names

    def test_lora_target_modules(self):
        cfg = TextEvasionConfig()
        assert isinstance(cfg.lora_target_modules, list)
        assert "q_proj" in cfg.lora_target_modules

    def test_independent_list_instances(self):
        cfg1 = TextEvasionConfig()
        cfg2 = TextEvasionConfig()
        cfg1.detector_names.append("extra")
        assert "extra" not in cfg2.detector_names


class TestMultiSPINConfig:
    def test_defaults(self):
        cfg = MultiSPINConfig()
        assert cfg.base_model == "Qwen/Qwen3.5-9B-Base"
        assert cfg.lambda_spin == 1.0
        assert cfg.num_iterations == 5
        assert cfg.steps_per_iteration == 1000
        assert cfg.monitor_style_embeddings is True

    def test_stylometric_features_list(self):
        cfg = MultiSPINConfig()
        assert len(cfg.stylometric_features) == 8
        assert "burstiness" in cfg.stylometric_features
        assert "ttr" in cfg.stylometric_features

    def test_override_spin_and_monitoring(self):
        cfg = MultiSPINConfig(lambda_spin=2.0, monitor_style_embeddings=False)
        assert cfg.lambda_spin == 2.0
        assert cfg.monitor_style_embeddings is False


class TestImageEvasionConfig:
    def test_defaults(self):
        cfg = ImageEvasionConfig()
        assert "stable-diffusion" in cfg.diffusion_model
        assert cfg.num_train_epochs == 50
        assert cfg.clip_range == pytest.approx(1e-4)
        assert cfg.per_prompt_stat_tracking is True

    def test_reward_weights_sum_to_one(self):
        cfg = ImageEvasionConfig()
        total = cfg.reward_evasion_weight + cfg.reward_clip_weight + cfg.reward_aesthetic_weight
        assert total == pytest.approx(1.0)

    def test_detector_names(self):
        cfg = ImageEvasionConfig()
        assert "efficientnet_b4" in cfg.detector_names


class TestArmsRaceConfig:
    def test_defaults(self):
        cfg = ArmsRaceConfig()
        assert cfg.num_rounds == 10
        assert cfg.modality == "text"
        assert cfg.defender_model == "roberta-base"
        assert cfg.use_meta_learning is False
        assert cfg.meta_second_order is True

    def test_eval_metrics_list(self):
        cfg = ArmsRaceConfig()
        assert "auroc" in cfg.eval_metrics
        assert "attack_success_rate" in cfg.eval_metrics

    def test_override_modality(self):
        cfg = ArmsRaceConfig(modality="image", num_rounds=5)
        assert cfg.modality == "image"
        assert cfg.num_rounds == 5

    def test_optional_configs_none(self):
        cfg = ArmsRaceConfig()
        assert cfg.attacker_config is None
        assert cfg.attacker_image_config is None
