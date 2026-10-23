"""Choose a physical microbatch without changing the effective GRPO batch."""

from copy import deepcopy
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from ..checkpointing import CheckpointPause, restore_rng, rng_state


def layout_identity(attacker, candidates):
    config = asdict(attacker.config)
    effective = config.pop("per_device_train_batch_size") * config.pop("gradient_accumulation_steps")
    return {"config": config, "effective_batch": effective, "candidates": list(candidates)}


def apply_layout(config, microbatch, effective):
    if microbatch < 1 or effective % microbatch:
        raise ValueError("Microbatch must divide the fixed effective batch")
    config.per_device_train_batch_size = microbatch
    config.gradient_accumulation_steps = effective // microbatch


def restore_layout(attacker, candidates):
    path = Path(attacker.config.output_dir) / "microbatch_profile.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record["identity"] != layout_identity(attacker, candidates):
            raise ValueError("Saved batch profile differs from the requested experiment")
        apply_layout(attacker.config, record["selected_microbatch"], record["identity"]["effective_batch"])


def apply_trainer_layout(trainer, config):
    if trainer.accelerator.num_processes != 1:
        raise ValueError("Batch profiling is restricted to one GPU")
    trainer.args.per_device_train_batch_size = config.per_device_train_batch_size
    trainer.args.gradient_accumulation_steps = config.gradient_accumulation_steps
    trainer.args.steps_per_generation = config.gradient_accumulation_steps
    trainer.args.generation_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
    trainer._train_batch_size = trainer.args.train_batch_size
    if trainer.accelerator.gradient_accumulation_steps != 1:
        raise ValueError("Transformers must manage accumulation without Accelerate dividing the loss again")


def select_measurement(measurements):
    successful = [row for row in measurements if row["status"] == "ok"]
    if not successful:
        raise RuntimeError("No profiled microbatch fits the H100 with the reserved memory headroom")
    return min(successful, key=lambda row: row["median_seconds"])["microbatch"]


def profile_inputs(attacker, effective):
    tokenizer, cfg = attacker.tokenizer, attacker.config
    prompts = tokenizer(attacker.prompts, padding=False)["input_ids"]
    count = effective // cfg.grpo_num_generations
    indices = sorted(range(len(prompts)), key=lambda index: len(prompts[index]), reverse=True)[:count]
    if len(indices) != count:
        raise ValueError("Not enough fitting prompts for the effective batch")
    width = max(map(len, prompts))
    completions = tokenizer([attacker.references[index] for index in indices], padding=False)["input_ids"]
    pad = tokenizer.pad_token_id
    prompt_ids = [[pad] * (width - len(prompts[index])) + prompts[index] for index in indices]
    completion_ids = [ids[:cfg.max_new_tokens] + [pad] * max(0, cfg.max_new_tokens - len(ids))
                      for ids in completions]
    device = next(attacker.model.parameters()).device
    values = {"prompt_ids": torch.tensor(prompt_ids, device=device),
              "completion_ids": torch.tensor(completion_ids, device=device)}
    values = {name: value.repeat_interleave(cfg.grpo_num_generations, dim=0) for name, value in values.items()}
    values.update(prompt_mask=torch.ones_like(values["prompt_ids"]),
                  completion_mask=torch.ones_like(values["completion_ids"]),
                  advantages=torch.linspace(-1, 1, cfg.grpo_num_generations, device=device).repeat(count),
                  num_items_in_batch=torch.tensor(effective * cfg.max_new_tokens, device=device))
    return values, indices


def profile_pass(trainer, inputs, microbatch):
    from trl.models.utils import disable_gradient_checkpointing
    from trl.trainer.utils import use_adapter

    trainer.model.zero_grad(set_to_none=True)
    ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
    mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
    with torch.no_grad(), trainer.accelerator.autocast(), disable_gradient_checkpointing(
        trainer.model, trainer.args.gradient_checkpointing_kwargs
    ), use_adapter(trainer.model, adapter_name="ref"):
        reference, _ = trainer._get_per_token_logps_and_entropies(
            trainer.model, ids, mask, inputs["completion_ids"].shape[1], batch_size=microbatch)
    size = len(ids)
    for start in range(0, size, microbatch):
        batch = {name: value[start:start + microbatch] if value.ndim else value for name, value in inputs.items()}
        batch["ref_per_token_logps"] = reference[start:start + microbatch]
        with trainer.accelerator.autocast(), trainer.compute_loss_context_manager():
            loss = trainer.compute_loss(trainer.model, batch)
        loss.backward()
    trainer.model.zero_grad(set_to_none=True)


def select_layout(attacker, trainer, candidates):
    from scripts.research_chain import write_record

    cfg = attacker.config
    path = Path(cfg.output_dir) / "microbatch_profile.json"
    if path.exists():
        apply_trainer_layout(trainer, cfg)
        return
    if trainer.loss_type != "dapo":
        raise ValueError("Layout profiling requires the current token-normalized DAPO loss")
    if not torch.cuda.is_available() or cfg.grpo_beta <= 0 or "ref" not in trainer.model.peft_config:
        raise ValueError("This layout profile requires CUDA and a fixed PEFT KL reference")
    identity = layout_identity(attacker, candidates)
    effective = identity["effective_batch"]
    if effective % cfg.grpo_num_generations or any(micro < 1 or effective % micro for micro in candidates):
        raise ValueError("Candidate layouts must preserve complete prompt groups and the effective batch")
    state = rng_state()
    training = trainer.model.training
    checkpointing = trainer.model.is_gradient_checkpointing
    metrics = deepcopy(trainer._metrics)
    measurements = []
    try:
        indices = list(range(min(8, len(attacker.prompts))))
        attacker.compute_reward([attacker.references[index] for index in indices],
                                [attacker.references[index] for index in indices])
        inputs, indices = profile_inputs(attacker, effective)
        trainer.model.train()
        trainer.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=trainer.args.gradient_checkpointing_kwargs)
        total_memory = torch.cuda.get_device_properties(next(trainer.model.parameters()).device).total_memory
        for microbatch in candidates:
            if getattr(attacker, "_stop_request", None) is not None and attacker._stop_request.requested:
                raise CheckpointPause("Batch profiling interrupted before any optimizer update")
            restore_rng(state)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            try:
                for repetition in range(3):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    profile_pass(trainer, inputs, microbatch)
                    torch.cuda.synchronize()
                    if repetition:
                        durations.append(time.perf_counter() - started)
                peak = torch.cuda.max_memory_allocated()
                status = "ok" if peak + 2 * 1024**3 <= total_memory else "insufficient_headroom"
                row = {"microbatch": microbatch, "status": status, "median_seconds": statistics.median(durations),
                       "timed_seconds": durations, "peak_allocated_gib": peak / 1024**3}
            except torch.OutOfMemoryError:
                row = {"microbatch": microbatch, "status": "out_of_memory"}
            finally:
                trainer.model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            measurements.append(row)
            print("GRPO fixed-batch layout profile", row, flush=True)
        selected = select_measurement(measurements)
        record = {"identity": identity, "selected_microbatch": selected, "measurements": measurements,
                  "effective_completions": effective, "prompt_groups": effective // cfg.grpo_num_generations,
                  "max_fitting_prompt_tokens": inputs["prompt_ids"].shape[1],
                  "completion_tokens": inputs["completion_ids"].shape[1],
                  "fitting_indices": indices, "optimizer_updates": 0,
                  "reserved_headroom_gib": 2, "scope": "reference forward plus GRPO forward/backward; generation and rewards excluded"}
        write_record(path, record)
        apply_layout(cfg, selected, effective)
        apply_trainer_layout(trainer, cfg)
    finally:
        trainer.model.zero_grad(set_to_none=True)
        if not checkpointing:
            trainer.model.gradient_checkpointing_disable()
        trainer.model.train(training)
        trainer._metrics = metrics
        restore_rng(state)
