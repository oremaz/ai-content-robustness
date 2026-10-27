"""Resource probes for detector adapters."""

from __future__ import annotations


def _check_hf_config(instance, model_name: str, trust_remote_code: bool = False) -> bool:
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        return True
    except Exception as error:
        instance._availability_error = str(error)
        return False

