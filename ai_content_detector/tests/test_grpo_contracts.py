from __future__ import annotations

from ai_content_detector.rl_evasion.text_evasion.grpo_trainer import GRPOTextEvasionTrainer


def test_compute_reward_uses_human_references_when_supplied():
    trainer = object.__new__(GRPOTextEvasionTrainer)
    seen = []

    def reward(source, completion):
        seen.append((source, completion))
        return {"total": 0.5}

    trainer.reward_fn = reward
    result = trainer.compute_reward(
        ["prompt"], ["completion"], references=["human continuation"],
    )
    assert result == [0.5]
    assert seen == [("human continuation", "completion")]
