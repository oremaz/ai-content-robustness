"""Native image architectures, independent of watermark loaders and model cards."""

from pathlib import Path
import sys

import torch


def use_backbones(path):
    sys.path.insert(0, str(Path(path)))
    sys.path.insert(0, str(Path(path) / "deps"))


def load_generator(spec, checkpoint=None, device="cuda"):
    from omegaconf import OmegaConf
    checkpoint = checkpoint or spec["checkpoint"]
    if spec["architecture"] == "rar":
        from deps.rar.modeling.rar import RAR
        model = RAR(OmegaConf.create(spec["config"]))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.set_random_ratio(0)
    elif spec["architecture"] == "uvit":
        from deps.efficientvit.diffusioncore.models.uvit import UViT
        config = OmegaConf.create({**spec["config"], "pretrained_path": checkpoint})
        model = UViT(config)
    else:
        raise ValueError("Unsupported native image architecture")
    return model.to(device).eval().requires_grad_(False)


def load_decoder(spec, device="cuda"):
    if spec["architecture"] == "rar":
        from deps.rar.modeling.titok import PretrainedTokenizer
        decoder = PretrainedTokenizer(spec["decoder"])
    else:
        from deps.efficientvit.models.efficientvit.dc_ae import DCAE, dc_ae_f64c128
        decoder = DCAE(dc_ae_f64c128("dc-ae-f64c128-in-1.0", spec["decoder"]))
    return decoder.to(device).eval().requires_grad_(False)


def state_for_generator(model, architecture):
    return model.get_trainable_modules().state_dict() if architecture == "uvit" else model.state_dict()


def restore_generator(model, state, architecture):
    target = model.get_trainable_modules() if architecture == "uvit" else model
    target.load_state_dict(state, strict=True)


def inject_lora(model, rank):
    from peft import LoraConfig, inject_adapter_in_model
    targets = [name for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
    return inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=32, target_modules=targets), model)
