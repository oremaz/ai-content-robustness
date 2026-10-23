"""Reusable measurements for unchanged evaluation texts and model states."""

import hashlib

import torch

from ..checkpointing import CheckpointPause, StopRequest, atomic_save, load_checkpoint


def semantic_values(encoder, pairs, directory, batch_size=16):
    from .protocol import digest

    signature = {"pairs": pairs, "encoder": encoder._model_name, "implementation": "batched_v1"}
    path = directory / (digest(signature) + ".pt")
    state = load_checkpoint(path, signature) or {"config": signature, "values": []}
    with StopRequest() as stop:
        for start in range(len(state["values"]), len(pairs), batch_size):
            if stop.requested:
                raise CheckpointPause("Batched semantic evaluation progress retained")
            batch = pairs[start:start + batch_size]
            valid = [row for row in batch if row["ai"].strip()]
            values = iter(encoder.batch([row["reference"] for row in valid], [row["ai"] for row in valid]))
            state["values"].extend(next(values) if row["ai"].strip() else None for row in batch)
            atomic_save(state, path)
    return [value for value in state["values"] if value is not None]


def evaluation_identity(attacker, pool, evaluation):
    from .workflow import checkpoint_identity
    from .protocol import digest

    weights = hashlib.sha256()
    for name, parameter in attacker.model.named_parameters():
        if parameter.requires_grad:
            weights.update(name.encode())
            weights.update(str((parameter.shape, parameter.dtype)).encode())
            weights.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    config = attacker.config
    return digest({"attacker": weights.hexdigest(), "defenders": checkpoint_identity(pool.checkpoints),
                   "evaluation": evaluation, "model": config.generator_model,
                   "max_seq_length": config.max_seq_length, "max_new_tokens": config.max_new_tokens,
                   "temperature": config.temperature, "top_p": config.top_p, "seed": 42,
                   "generation_batch_size": getattr(attacker, "generation_batch_size", 1)})
