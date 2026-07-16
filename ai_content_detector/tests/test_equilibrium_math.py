"""Protocol tests for paired arms-race update effects."""

from __future__ import annotations

import pytest

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
