"""GPU validation before the longer watermark jobs become eligible."""

import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .protocol import write_json


def offline(manifest):
    from distseal.loader import load_generator, load_decoder, load_detector, parse_model_card
    from .image import cards
    result = {}
    for family in ("rar", "dcae"):
        generator_card, decoder_card, detector_card = cards(manifest, family)
        decoder = load_decoder(decoder_card, device="cpu")
        result[decoder_card] = sum(p.numel() for p in decoder.parameters())
        del decoder
        gc.collect()
        detector, message = load_detector(detector_card, device="cpu")
        checkpoint = torch.load(parse_model_card(detector_card).checkpoint,
                                map_location="cpu", weights_only=True, mmap=True)["model"]
        detector.load_state_dict({key.removeprefix("detector."): value for key,value in checkpoint.items()
                                  if key.startswith("detector.")}, strict=True)
        result[detector_card] = {"bits": message.numel(), "strict_state": True}
        del detector, checkpoint
        gc.collect()
        for watermarked in (True, False):
            card = cards(manifest, family, watermarked)[0]
            model = load_generator(card, device="cpu")
            result[card] = sum(p.numel() for p in model.parameters())
            del model
            gc.collect()
        print("Offline loaders verified", family, flush=True)
    write_json(Path(manifest["directory"]) / "cpu_preflight.json", result)
    return result


def run(manifest):
    from .text import configured_cache, distillation_loss, load_model, watermark_config
    import textseal.watermarking.generator as official
    from textseal.watermarking.detector import localized_detect
    official._make_cache = configured_cache
    root = Path(manifest["directory"])
    data = json.loads(Path(manifest["data"]).read_text())
    result = {}
    teacher, tokenizer = load_model(manifest["qwen"]["path"])
    prompt = tokenizer.decode(data["text"]["train"][0]["tokens"][:50])
    wm = watermark_config(manifest, "textseal")
    texts = official.TextSealGenerator(teacher, tokenizer, wm).generate([prompt], max_gen_len=8)
    detection = localized_detect(texts[0], tokenizer, wm, model=teacher)
    result["textseal_generation_and_detection"] = {"tokens": len(tokenizer.encode(texts[0])),
                                                   "p_value": detection.final_pvalue}
    student, _ = load_model(manifest["qwen"]["path"], training=True)
    ids = torch.tensor([data["text"]["train"][0]["tokens"][:64]], device="cuda")
    with torch.no_grad():
        target = teacher(ids, use_cache=False).logits[:, :-1]
    logits = student(ids, use_cache=False).logits[:, :-1]
    loss = distillation_loss(logits, target, ids[:, :-1, None], torch.ones_like(ids[:, :-1], dtype=torch.bool),
                             watermark_config(manifest, "greenlist"))
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Qwen distillation produced invalid gradients")
    result["qwen_distillation_backward"] = {"kl": float(loss.detach()), "gradient_norm": float(norm)}
    del loss, logits, target, student, teacher
    gc.collect(); torch.cuda.empty_cache()
    from distseal.loader import load_generator, load_decoder, load_detector, parse_model_card
    from distseal.modules.lora_helper import apply_lora_to_linear, mark_only_lora_as_trainable
    from deps.rar.modeling.modules.losses import ARLoss
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
    from PIL import Image
    from .image import cards, bit_scores
    for family in ("rar", "dcae"):
        generator_card, decoder_card, detector_card = cards(manifest, family)
        generator = load_generator(generator_card, device="cuda")
        decoder = load_decoder(decoder_card, device="cuda")
        detector, message = load_detector(detector_card, device="cuda")
        state = torch.load(parse_model_card(detector_card).checkpoint,
                           map_location="cpu", weights_only=True, mmap=True)["model"]
        detector.load_state_dict({k.removeprefix("detector."):v for k,v in state.items()
                                  if k.startswith("detector.")}, strict=True)
        del state
        base = load_generator(cards(manifest, family, watermarked=False)[0], device="cpu")
        if {k:tuple(p.shape) for k,p in base.named_parameters()} != {k:tuple(p.shape) for k,p in generator.named_parameters()}:
            raise ValueError("Original and watermarked generator parameters do not align")
        del base
        gc.collect()
        labels = torch.zeros(1, dtype=torch.long, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if family == "rar":
                latent = generator.generate(labels, guidance_scale=6.9, guidance_scale_pow=1.5, randomize_temperature=1.02)
                images = decoder.decode(latent)
            else:
                latent = generator.generate(labels.int(), torch.full_like(labels, 1000).int(), scale=1.)
                images = ((decoder.decode(latent/.2889).float()*127.5+128)/255).clamp(0,1)
        scores = bit_scores(detector, message, images.float())
        apply_lora_to_linear(generator, r=16, lora_alpha=32, target_linear_modules=("",))
        parameters = list(mark_only_lora_as_trainable(generator))
        generator.train()
        size = 256 if family == "rar" else 512
        with Image.open(data["images"]["train"][0]) as image:
            tensor = Compose([Resize(size), CenterCrop(size), ToTensor()])(image.convert("RGB")).unsqueeze(0).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            latent = decoder.encode(tensor if family == "rar" else tensor*2-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if family == "rar":
                generator.disable_kv_cache()
                logits, target = generator(latent, generator.preprocess_condition(labels), return_labels=True)
                loss, _ = ARLoss(generator.config)(logits.float(), target)
            else:
                loss, _ = generator(latent*.2889, labels.int())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError(f"Invalid {family} forgetting gradients")
        result[family] = {"bit_accuracy": scores[0], "training_loss": float(loss.detach()), "gradient_norm": float(norm)}
        print("GPU preflight", family, result[family], flush=True)
        del loss, generator, decoder, detector, latent, images, parameters
        if family == "rar":
            del logits, target
        gc.collect(); torch.cuda.empty_cache()
    write_json(root / "gpu_preflight.json", result)
