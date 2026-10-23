"""Validate a retained attacker and its cached evaluations before defender updates."""

import hashlib
import json
from pathlib import Path


def verify(bundle):
    checkpoint = Path(bundle["checkpoint"])
    required = {str(checkpoint / name) for name in
                ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json")}
    required.update(bundle[key] for key in ("pairs", "before", "after_attack"))
    if not required <= set(bundle["sha256"]):
        raise ValueError("Warm-start identity is missing required artifacts")
    for name, expected in bundle["sha256"].items():
        with Path(name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Warm-start artifact changed: {name}")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state["global_step"] != bundle["updates"]:
        raise ValueError("Warm-start update count differs from its saved checkpoint")


def load_policy(attacker, bundle):
    from peft import load_peft_weights, set_peft_model_state_dict

    weights = load_peft_weights(bundle["checkpoint"], device="cpu")
    result = set_peft_model_state_dict(attacker.model, weights)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise ValueError("Retained attacker adapter does not match the generator")


def read(bundle, phase):
    return json.loads(Path(bundle[phase]).read_text())
