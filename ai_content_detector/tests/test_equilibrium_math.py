"""Protocol tests for paired arms-race update effects."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ai_content_detector.rl_evasion.config import ArmsRaceConfig, TextEvasionConfig
from ai_content_detector.rl_evasion.arms_race.equilibrium import ArmsRaceExperiment


def test_update_effects_use_paired_differences():
    effects = ArmsRaceExperiment._compute_update_effects(
        before_attack={"attack_success_rate": 0.2},
        after_attack={"attack_success_rate": 0.6, "accuracy": 0.4, "auroc": 0.5},
        after_defense={"accuracy": 0.75, "auroc": 0.8},
    )
    assert effects["attacker_success_change"] == pytest.approx(0.4)
    assert effects["defender_accuracy_change"] == pytest.approx(0.35)
    assert effects["defender_auroc_change"] == pytest.approx(0.3)


def test_update_effects_preserve_negative_changes():
    effects = ArmsRaceExperiment._compute_update_effects(
        before_attack={"attack_success_rate": 0.8},
        after_attack={"attack_success_rate": 0.5, "accuracy": 0.7, "auroc": 0.9},
        after_defense={"accuracy": 0.6, "auroc": 0.75},
    )
    assert effects["attacker_success_change"] == pytest.approx(-0.3)
    assert effects["defender_accuracy_change"] == pytest.approx(-0.1)
    assert effects["defender_auroc_change"] == pytest.approx(-0.15)


def test_image_mode_fails_before_loading_an_attacker():
    experiment = ArmsRaceExperiment()
    with pytest.raises(NotImplementedError, match="adaptive image defender"):
        experiment._setup_image()


def test_partition_removes_eval_prompts_from_attacker_training():
    class Attacker:
        prompts = [f"p{i}" for i in range(10)]
        references = [f"r{i}" for i in range(10)]

    experiment = ArmsRaceExperiment()
    experiment.attacker = Attacker()
    experiment._partition_prompts(eval_frac=0.2)

    assert len(experiment.eval_prompts) == 2
    assert len(experiment.attacker.prompts) == 8
    assert set(experiment.attacker.prompts).isdisjoint(experiment.eval_prompts)
    assert set(experiment.attacker.references).isdisjoint(experiment.eval_references)


def test_partition_reuses_attacker_held_out_split():
    experiment = ArmsRaceExperiment()
    experiment.attacker = SimpleNamespace(
        prompts=["train"], references=["human train"],
        eval_prompts=["eval"], eval_references=["human eval"],
    )
    experiment._partition_prompts()
    assert experiment.train_prompts == ["train"]
    assert experiment.eval_prompts == ["eval"]
    assert experiment.eval_references == ["human eval"]


def test_warmup_uses_training_pairs_and_connects_defender(tmp_path):
    experiment = ArmsRaceExperiment(ArmsRaceConfig(
        output_dir=str(tmp_path), defender_retrain_samples=4,
    ))
    experiment.train_prompts = ["p0", "p1", "p2"]
    experiment.train_references = ["h0", "h1", "h2"]
    experiment.eval_prompts = ["held out"]
    experiment._generate = Mock(return_value=["a0", "a1"])
    experiment.defender = SimpleNamespace(
        retrain=Mock(return_value={"round": 0, "accuracy": 0.75}),
        predict=Mock(return_value=0.8),
    )
    experiment.attacker = SimpleNamespace(
        train=Mock(), reward_fn=SimpleNamespace(detector_rewards=[]),
    )

    experiment._warmup_defender()

    experiment._generate.assert_called_once_with(["p0", "p1"], seed=43)
    experiment.defender.retrain.assert_called_once_with(
        ai_texts=["a0", "a1"], human_texts=["h0", "h1"], warmup=True,
    )
    experiment.attacker.train.assert_not_called()
    assert experiment.attacker.reward_fn.detector_rewards[0].name == "adaptive_defender"
    assert experiment.history == []
    assert json.loads((tmp_path / "defender_warmup.json").read_text())["round"] == 0


def test_warmup_rejects_insufficient_training_data():
    experiment = ArmsRaceExperiment()
    experiment.train_prompts = ["p0"]
    with pytest.raises(ValueError, match="at least two"):
        experiment._warmup_defender()


def test_text_config_defines_evaluation_fraction():
    assert TextEvasionConfig().evaluation_fraction == 0.2
