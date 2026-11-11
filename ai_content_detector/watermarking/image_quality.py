"""Matched image quality evaluation of saved watermark attack outputs."""

from pathlib import Path

from PIL import Image

from .closed_image import file_hash


METRICS = ("lpips", "ssim", "psnr", "clip_image_cosine", "clip_alignment", "reference_clip_alignment")


def align_by_index(reference, candidate):
    indices = [r["index"] for r in reference]
    indexed = {r["index"]: r for r in candidate}
    if len(set(indices)) != len(indices) or len(indexed) != len(candidate):
        raise ValueError("Duplicate image indices")
    if set(indices) != set(indexed):
        raise ValueError("Candidate images differ from reference indices")
    result = [indexed[index] for index in indices]
    if any(a["label"] != b["label"] for a, b in zip(reference, result)):
        raise ValueError("Candidate class labels differ from the paired reference")
    return result


def read_pair(row):
    images = []
    for name in ("reference", "candidate"):
        path = Path(row[name])
        if file_hash(path) != row[name+"_sha256"]:
            raise ValueError(f"Frozen {name} PNG changed: {row['index']}")
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    if images[0].size != images[1].size:
        raise ValueError("Paired images have different native resolutions")
    return images


def paired_psnr(first, second):
    if first.shape != second.shape or first.ndim != 4:
        raise ValueError("PSNR requires matching NCHW image batches")
    return -10*(first-second).square().flatten(1).mean(-1).clamp_min(1e-12).log10()


class ImageQuality:
    def __init__(self, clip_path, metric_cache, device="cuda"):
        import torch
        import lpips
        from torchvision.models import ResNet50_Weights
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.torch = torch
        torch.hub.set_dir(metric_cache)
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
        self.clip = CLIPModel.from_pretrained(clip_path, local_files_only=True).to(device).eval().requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
        self.categories = ResNet50_Weights.DEFAULT.meta["categories"]
        features = []
        with torch.inference_mode():
            for start in range(0, len(self.categories), 128):
                texts = ["a photo of a "+s for s in self.categories[start:start+128]]
                tokens = self.processor(text=texts, padding=True, return_tensors="pt").to(device)
                value = self.clip.get_text_features(**tokens)
                value = getattr(value, "pooler_output", value)
                features.append(torch.nn.functional.normalize(value.float(), dim=-1))
        self.text_features = torch.cat(features)

    def score(self, originals, candidates, labels):
        import torch
        import torch.nn.functional as F
        from torchvision.transforms.functional import to_tensor
        from pytorch_msssim import ssim
        if not len(originals) == len(candidates) == len(labels) or not originals:
            raise ValueError("Image and class batches must match")
        if any(not 0 <= label < len(self.categories) for label in labels):
            raise ValueError("ImageNet label is outside the canonical class ordering")
        if any(a.size != b.size for a, b in zip(originals, candidates)):
            raise ValueError("Native paired resolutions must match")
        a = torch.stack([to_tensor(x) for x in originals]).to(self.device)
        b = torch.stack([to_tensor(x) for x in candidates]).to(self.device)
        with torch.inference_mode():
            perceptual = self.lpips(a*2-1, b*2-1).flatten()
            similarity = ssim(a, b, data_range=1., size_average=False)
            psnr = paired_psnr(a, b)
            pixels = self.processor(images=originals+candidates, return_tensors="pt").to(self.device)
            features = self.clip.get_image_features(**pixels)
            features = F.normalize(getattr(features, "pooler_output", features).float(), dim=-1)
            n = len(originals)
            text = self.text_features[torch.tensor(labels, device=self.device)]
            values = {"lpips": perceptual, "ssim": similarity, "psnr": psnr,
                      "clip_image_cosine": (features[:n]*features[n:]).sum(-1).clamp(-1, 1),
                      "clip_alignment": (features[n:]*text).sum(-1).clamp(-1, 1),
                      "reference_clip_alignment": (features[:n]*text).sum(-1).clamp(-1, 1)}
        return [{key: float(value[i]) for key, value in values.items()} for i in range(n)]
