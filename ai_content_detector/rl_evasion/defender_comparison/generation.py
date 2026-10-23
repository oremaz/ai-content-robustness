"""Checkpointed batched sampling for the comparison races."""

from pathlib import Path

import torch

from ..checkpointing import (CheckpointPause, StopRequest, atomic_save, evaluation_state,
                             restore_rng, rng_state)


def generate_batched(attacker, prompts, directory, context, seed=42, batch_size=8):
    from .protocol import digest

    if batch_size < 1:
        raise ValueError("Generation batch size must be positive")
    signature = {"prompts": prompts, "seed": seed, "context": context,
                 "model": attacker.config.generator_model, "batch_size": batch_size,
                 "max_seq_length": attacker.config.max_seq_length,
                 "max_new_tokens": attacker.config.max_new_tokens,
                 "temperature": attacker.config.temperature, "top_p": attacker.config.top_p,
                 "implementation": "left_padded_batches_v1"}
    path = Path(directory) / "generation" / (digest(signature) + ".pt")
    caller_rng = rng_state()
    training = attacker.model.training
    padding_side = attacker.tokenizer.padding_side
    device = next(attacker.model.parameters()).device
    try:
        torch.manual_seed(seed)
        state = evaluation_state(path, signature, attacker.model)
        attacker.model.eval()
        attacker.tokenizer.padding_side = "left"
        with StopRequest() as stop:
            for start in range(len(state["generated"]), len(prompts), batch_size):
                if stop.requested:
                    raise CheckpointPause("Batched race generation progress retained")
                tokens = attacker.tokenizer(prompts[start:start + batch_size], return_tensors="pt",
                                            padding=True, truncation=True,
                                            max_length=attacker.config.max_seq_length).to(device)
                with torch.no_grad():
                    output = attacker.model.generate(
                        **tokens, max_new_tokens=attacker.config.max_new_tokens,
                        temperature=attacker.config.temperature, top_p=attacker.config.top_p,
                        do_sample=True, pad_token_id=attacker.tokenizer.pad_token_id)
                texts = attacker.tokenizer.batch_decode(output[:, tokens["input_ids"].shape[1]:],
                                                        skip_special_tokens=True)
                if len(texts) != min(batch_size, len(prompts) - start):
                    raise ValueError("Generated texts and prompt batch are not aligned")
                state["generated"].extend(texts)
                state["rng"] = rng_state()
                atomic_save(state, path)
        return state["generated"]
    finally:
        attacker.tokenizer.padding_side = padding_side
        attacker.model.train(training)
        restore_rng(caller_rng)
