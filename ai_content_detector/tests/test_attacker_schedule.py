from dataclasses import asdict

import pytest
import torch

from ai_content_detector.rl_evasion.checkpointing import atomic_save
from ai_content_detector.rl_evasion.config import TextEvasionConfig
from ai_content_detector.rl_evasion.text_evasion.grpo_trainer import GRPOTextEvasionTrainer


def test_constant_warmup_reaches_requested_rate_and_resumes_without_rewarming():
    from transformers import get_scheduler

    def make():
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=5e-5)
        scheduler = get_scheduler("constant_with_warmup", optimizer, num_warmup_steps=50,
                                  num_training_steps=300)
        return optimizer, scheduler

    optimizer, scheduler = make()
    rates = [scheduler.get_last_lr()[0]]
    saved = None
    for step in range(1, 301):
        optimizer.step()
        scheduler.step()
        rates.append(scheduler.get_last_lr()[0])
        if step == 137:
            import copy
            saved = copy.deepcopy((optimizer.state_dict(), scheduler.state_dict()))
    assert rates[0] == 0.0
    assert rates[1] == pytest.approx(1e-6)
    assert rates[25] == pytest.approx(2.5e-5)
    assert rates[49] == pytest.approx(4.9e-5)
    assert rates[50:] == pytest.approx([5e-5] * 251)
    resumed_optimizer, resumed_scheduler = make()
    resumed_optimizer.load_state_dict(saved[0])
    resumed_scheduler.load_state_dict(saved[1])
    resumed_rates = [resumed_scheduler.get_last_lr()[0]]
    for step in range(138, 301):
        resumed_optimizer.step()
        resumed_scheduler.step()
        resumed_rates.append(resumed_scheduler.get_last_lr()[0])
    assert resumed_rates == rates[137:]


def test_restart_accepts_legacy_defaults_but_rejects_a_changed_schedule(tmp_path):
    config = TextEvasionConfig(output_dir=str(tmp_path))
    legacy = asdict(config)
    legacy.pop("lr_scheduler_type")
    legacy.pop("warmup_steps")
    path = tmp_path / "restart_context.pt"
    atomic_save({"config": legacy, "max_steps": 500}, path)
    assert GRPOTextEvasionTrainer(config)._load_restart_context(path)["max_steps"] == 500
    changed = TextEvasionConfig(output_dir=str(tmp_path), learning_rate=5e-5,
                               lr_scheduler_type="constant_with_warmup", warmup_steps=50)
    with pytest.raises(ValueError, match="configuration differs"):
        GRPOTextEvasionTrainer(changed)._load_restart_context(path)
    with pytest.raises(ValueError, match="require TRL"):
        GRPOTextEvasionTrainer(changed)._train_group_normalized_reinforce(max_steps=300)
