"""Preserve both the trainable policy and fixed PEFT reference on restart."""

import hashlib
import json
from pathlib import Path

import torch

def reference_identity(model):
    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if ".ref." in name:
            if parameter.requires_grad:
                raise ValueError("KL reference parameters must remain frozen")
            digest.update(name.encode())
            digest.update(str((tuple(parameter.shape), parameter.dtype)).encode())
            value = parameter.detach().cpu().contiguous().view(-1).view(torch.uint8)
            digest.update(memoryview(value.numpy()))
            count += parameter.numel()
    if not count:
        raise ValueError("KL requires a frozen reference adapter")
    return {"sha256": digest.hexdigest(), "parameters": count}


def configure_reference_restart(trainer, config):
    if not config.grpo_beta:
        return
    from scripts.research_chain import write_record

    path = Path(config.output_dir) / "reference_identity.json"
    identity = reference_identity(trainer.model)
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("KL reference differs from the original run")
    else:
        write_record(path, identity)

    def load(checkpoint, model=None):
        target = model if model is not None else trainer.model
        reference = Path(checkpoint) / "ref"
        if not any((reference / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            raise ValueError("Native checkpoint is missing its frozen KL reference")
        target.load_adapter(str(reference), "ref", is_trainable=False, autocast_adapter_dtype=False)
        target.load_adapter(str(checkpoint), "default", is_trainable=True)
        target.set_adapter("default")
        if reference_identity(target) != identity:
            raise ValueError("Native checkpoint changed the frozen KL reference")

    trainer._load_from_checkpoint = load
