"""Resource probes and image conversion for detector adapters."""

from __future__ import annotations


import numpy as np

from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _check_hf_image_config(instance, model_name: str) -> bool:
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_name)
        return True
    except Exception as error:
        instance._availability_error = str(error)
        return False


def _pil_to_tensor(image: Image.Image, size: int = 200) -> "np.ndarray":
    """Convert a PIL image to a normalized tensor (C, H, W) as float32 numpy."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(image.convert("RGB"))
    return tensor.unsqueeze(0).numpy().astype(np.float32)

