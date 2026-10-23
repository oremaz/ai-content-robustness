"""Attacker learning-rate schedules with native optimizer-step resumption."""

import math

from torch.optim.lr_scheduler import LambdaLR


def warmup_stable_cosine(optimizer, *, num_training_steps, num_warmup_steps,
                         num_decay_steps, min_lr_ratio):
    if not 0 <= num_warmup_steps < num_training_steps - num_decay_steps:
        raise ValueError("Warm-up must finish before cosine decay begins")
    if not 0 < num_decay_steps <= num_training_steps or not 0 <= min_lr_ratio <= 1:
        raise ValueError("Invalid cosine decay length or minimum learning-rate ratio")
    decay_start = num_training_steps - num_decay_steps

    def factor(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        if step <= decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / num_decay_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2

    return LambdaLR(optimizer, factor)


def configure_scheduler(trainer, config):
    if config.lr_scheduler_type != "warmup_stable_decay":
        return

    def create_scheduler(num_training_steps, optimizer=None):
        if trainer.lr_scheduler is None:
            trainer.lr_scheduler = warmup_stable_cosine(
                optimizer if optimizer is not None else trainer.optimizer,
                num_training_steps=num_training_steps, num_warmup_steps=config.warmup_steps,
                **config.lr_scheduler_kwargs,
            )
            trainer._created_lr_scheduler = True
        return trainer.lr_scheduler

    trainer.create_scheduler = create_scheduler
