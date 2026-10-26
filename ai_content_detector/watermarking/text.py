"""TextSeal and fixed-key distribution distillation into ordinary Qwen weights."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..rl_evasion.checkpointing import (CheckpointClock, CheckpointPause, atomic_save,
                                      model_state, restore_model, rng_state, restore_rng)
from ..rl_evasion.config import seed_everything
from .protocol import detection_metrics, digest, operating_point, write_json


def configured_cache(model):
    from transformers.cache_utils import DynamicCache
    return DynamicCache(config=model.config)


def load_model(path, training=False, training_mode="lora"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                                               dtype=torch.float32 if training and training_mode == "full" else torch.bfloat16,
                                               attn_implementation="sdpa").cuda()
    model.eval().requires_grad_(False)
    if training:
        if training_mode == "lora":
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0,
                                                   target_modules="all-linear", task_type="CAUSAL_LM"))
        elif training_mode == "full":
            model.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {training_mode}")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.train()
    return model, tokenizer


def watermark_config(manifest, kind):
    from textseal.watermarking.config import WatermarkConfig
    return WatermarkConfig(watermark_type=kind, secret_key=manifest["key"],
                           ngram=3 if kind == "textseal" else 1,
                           mixing_alpha=.9, gamma=.25, delta=2., after_topp=False,
                           method="binary" if kind == "greenlist" else "uniform")


def distillation_loss(student_logits, teacher_logits, contexts, mask, config, chunk=32):
    from textseal.watermarking.core import prf_binary
    student_logits = student_logits.reshape(-1, student_logits.shape[-1])
    teacher_logits = teacher_logits.reshape_as(student_logits)
    contexts, mask = contexts.reshape(-1, config.ngram), mask.reshape(-1).bool()
    count = mask.sum()
    if count == 0:
        raise ValueError("Distillation requires at least one valid prediction")
    total = student_logits.new_zeros((), dtype=torch.float32)
    vocab = torch.arange(student_logits.shape[-1], device=student_logits.device)
    for start in range(0, len(mask), chunk):
        valid = mask[start:start+chunk]
        if not valid.any():
            continue
        with torch.no_grad():
            windows = contexts[start:start+chunk][valid]
            green = prf_binary(windows, vocab.expand(len(windows), -1), config.secret_key, config.gamma)
            target = F.softmax(teacher_logits[start:start+chunk][valid].float() + config.delta * green, dim=-1)
        logp = F.log_softmax(student_logits[start:start+chunk][valid].float(), dim=-1)
        total = total + F.kl_div(logp, target, reduction="sum")
    return total / count


def forward_kl(student_logits, teacher_logits, mask, chunk=32):
    student = student_logits.reshape(-1, student_logits.shape[-1])
    teacher = teacher_logits.detach().reshape_as(student)
    valid = mask.reshape(-1).bool()
    if not valid.any():
        raise ValueError("Distillation requires a valid prediction")
    loss = student.new_zeros((), dtype=torch.float32)
    for start in range(0, len(valid), chunk):
        selected = valid[start:start+chunk]
        if selected.any():
            target = teacher[start:start+chunk][selected].float().softmax(-1)
            logp = student[start:start+chunk][selected].float().log_softmax(-1)
            loss = loss + F.kl_div(logp, target, reduction="sum")
    return loss / valid.sum()


def fit(manifest, stop, clean=False):
    root = Path(manifest["directory"])
    name = manifest.get("clean_stage", "text_clean") if clean else manifest.get("distillation_stage", "distill")
    output = root / name
    output.mkdir(exist_ok=True)
    data = json.loads(Path(manifest["data"]).read_text())["text"]
    config = manifest["text_training"]
    steps = config["clean_steps"] if clean else config["steps"]
    source = manifest.get("clean_source", str(root / manifest.get("distilled_model", "distill/model"))) if clean else manifest["qwen"]["path"]
    mode = config.get("mode", "lora")
    model, tokenizer = load_model(source, training=True, training_mode=mode)
    teacher = None
    if not clean:
        teacher, _ = load_model(source)
    latent = manifest.get("method") == "distseal_latent_text"
    handle = None
    if not clean and latent:
        from .latent_text import load_pair, install_embedder
        embedder, _, _ = load_pair(manifest)
        handle = install_embedder(teacher, embedder, manifest["latent_training"]["message"])
    wm_config = watermark_config(manifest, "greenlist") if not clean and not latent else None
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["clean_lr"] if clean else config["lr"], weight_decay=0., foreach=False)
    signature = {"protocol": manifest["protocol_sha256"], "clean": clean, "source": source}
    step, tokens_seen, history = 0, 0, []
    restart = output / "restart.pt"
    if restart.exists():
        state = torch.load(restart, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Text training restart does not match protocol")
        restore_model(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        step, tokens_seen, history = state["step"], state["tokens_seen"], state["history"]
        del state
    clock = CheckpointClock(600)
    rows = data["removal" if clean else "train"]
    def save():
        atomic_save({"signature": signature, "model": model_state(model), "optimizer": optimizer.state_dict(),
                     "rng": rng_state(), "step": step, "tokens_seen": tokens_seen, "history": history}, restart)
        write_json(output / "progress.json", {"step": step, "tokens_seen": tokens_seen, "history": history,
                                              "uses_original_weights_for_attack": False if clean else None})
        clock.saved()
    def export_intermediate():
        if clean and step in config.get("export_steps", []) and step != steps:
            destination = output / f"checkpoint-{step}"
            model.save_pretrained(destination, max_shard_size="2GB")
            tokenizer.save_pretrained(destination)
            write_json(destination / "complete.json", {"step": step, "mode": mode})
    export_intermediate()
    while step < steps:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        lr = (config["clean_lr"] if clean else config["lr"]) * min(1., (step+1)/100)
        lr *= .1 + .9 * .5 * (1 + math.cos(math.pi * step / steps))
        for group in optimizer.param_groups:
            group["lr"] = lr
        indices = np.random.default_rng(manifest["seed"] + step).integers(len(rows), size=config["accumulation"])
        for index in indices:
            ids = torch.tensor([rows[index]["tokens"][:config["length"]]], device="cuda")
            if clean:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            else:
                with torch.no_grad():
                    target = teacher(input_ids=ids, use_cache=False).logits[:, :-1].detach()
                logits = model(input_ids=ids, use_cache=False).logits[:, :-1]
                # Logit at position t predicts token t+1, seeded by token t.
                mask = torch.ones_like(ids[:, :-1], dtype=torch.bool)
                loss = (forward_kl(logits, target, mask) if latent else
                        distillation_loss(logits, target, ids[:, :-1, None], mask, wm_config))
                del target, logits
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite watermark training loss")
            (loss / config["accumulation"]).backward()
            loss_sum += float(loss.detach()) / config["accumulation"]
            tokens_seen += ids.numel()-1
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite text training gradient")
        optimizer.step()
        step += 1
        if step % 50 == 0:
            history.append({"step": step, "loss": loss_sum, "tokens_seen": tokens_seen, "lr": lr})
            print(name, history[-1], flush=True)
        if step % 500 == 0 or clock.due() or stop.requested or step == steps:
            save()
        export_intermediate()
        if stop.requested:
            raise CheckpointPause(f"{name} checkpoint saved at step {step}")
    if handle is not None:
        handle.remove()
    del teacher, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    model = (model.merge_and_unload() if mode == "lora" else model).eval()
    model.config.use_cache = True
    model.save_pretrained(output / "model", max_shard_size="2GB")
    tokenizer.save_pretrained(output / "model")
    write_json(output / "complete.json", {"steps": step, "tokens_seen": tokens_seen,
                "merged": True, "architecture_changed": False, "ordinary_sampler_at_inference": True,
                "source": source, "original_weights_accessed_during_attack": False if clean else None})


def generate(model, tokenizer, rows, output, config, stop, wm=None):
    import textseal.watermarking.generator as official_generator
    from textseal.watermarking.generator import TextSealGenerator, WmGenerator, GreenlistGenerator
    # Current Transformers represents Qwen's hybrid state with a configured
    # DynamicCache; the upstream fallback constructs an attention-only cache.
    official_generator._make_cache = configured_cache
    output = Path(output)
    result = json.loads(output.read_text()) if output.exists() else []
    expected = digest([r["id"] for r in rows])
    if result and result[0]["partition"] != expected:
        raise ValueError("Generation cache uses different prompts")
    sampler = TextSealGenerator if wm and wm.watermark_type == "textseal" else GreenlistGenerator if wm else WmGenerator
    if wm is None:
        from textseal.watermarking.config import WatermarkConfig
        wm = WatermarkConfig(watermark_type="none")
    generator = sampler(model, tokenizer, wm)
    for start in range(len(result), len(rows), config["batch"]):
        batch = rows[start:start+config["batch"]]
        prompts = [tokenizer.decode(row["tokens"][:50]) for row in batch]
        seed_everything(config["seed"] + start)
        texts = generator.generate(prompts, max_gen_len=config.get("max_tokens", 256),
                                   temperature=config.get("temperature", .8), top_p=config.get("top_p", .95))
        result.extend({"id": row["id"], "partition": expected, "prompt": prompt, "text": text,
                       "tokens": tokenizer.encode(text, add_special_tokens=False)}
                      for row, prompt, text in zip(batch, prompts, texts))
        write_json(output, result)
        if stop.requested:
            raise CheckpointPause("Text generation saved")
    return result


def score_texts(tokenizer, texts, wm, proxy=None):
    from scipy.stats import binom
    from textseal.watermarking.core import prf_binary
    from textseal.watermarking.detector import localized_detect
    scores, records = [], []
    for text in texts:
        if wm.watermark_type == "textseal":
            # The upstream global detector takes min(weighted,unweighted) without
            # correcting it. The localized ensemble supplies its own correction.
            result = localized_detect(text, tokenizer, wm, model=proxy)
            p = float(result.final_pvalue)
        else:
            ids = tokenizer.encode(text, add_special_tokens=False)
            pairs = sorted(set(zip(ids[:-1], ids[1:])))
            if not pairs:
                p = 1.
            else:
                contexts = torch.tensor([[a] for a, _ in pairs])
                values = torch.tensor([b for _, b in pairs])
                green = prf_binary(contexts, values[:, None], wm.secret_key, wm.gamma)
                p = float(binom.sf(int(green.sum())-1, len(pairs), wm.gamma))
        scores.append(-math.log10(max(p, 1e-300)))
        records.append({"p_value": p, "score": scores[-1]})
    return scores, records


@torch.no_grad()
def quality(model, rows):
    nll, count = 0., 0
    for row in rows[:64]:
        ids = torch.tensor([row["tokens"][:320]], device="cuda")
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        nll += float(loss) * (ids.numel()-1)
        count += ids.numel()-1
    return {"human_holdout_nll": nll/count, "human_holdout_perplexity": math.exp(nll/count),
            "quality_documents": min(64, len(rows))}


@torch.no_grad()
def generation_perplexity(reference, tokenizer, sampled):
    nll, count = 0., 0
    for row in sampled:
        tokens = row["tokens"]
        if not tokens:
            continue
        prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
        ids = torch.tensor([prompt + tokens], device="cuda")
        labels = ids.clone()
        labels[:, :len(prompt)] = -100
        loss = reference(input_ids=ids, labels=labels, use_cache=False).loss
        nll += float(loss) * len(tokens)
        count += len(tokens)
    return {"reference_generation_perplexity": math.exp(nll/count) if count else None,
            "reference_scored_tokens": count}


def evaluate(manifest, stage, stop):
    root = Path(manifest["directory"])
    output = root / stage
    output.mkdir(exist_ok=True)
    data = json.loads(Path(manifest["data"]).read_text())["text"]
    wm = watermark_config(manifest, "textseal" if stage == "textseal" else "greenlist")
    source = manifest["qwen"]["path"] if stage == "textseal" else str(root / "distill/model")
    if stage == "text_clean":
        fit(manifest, stop, clean=True)
        source = str(root / "text_clean/model")
    model, tokenizer = load_model(source)
    generation = {"batch": 8, "seed": manifest["seed"]}
    calibration = [tokenizer.decode(row["tokens"][50:306]) for row in data["validation"]]
    human = [tokenizer.decode(row["tokens"][50:306]) for row in data["test"]]
    proxy = model if stage == "textseal" else None
    null_cal, _ = score_texts(tokenizer, calibration, wm, proxy)
    threshold = operating_point(null_cal)
    null_test, _ = score_texts(tokenizer, human, wm, proxy)
    results = {"method": wm.watermark_type, "threshold_calibration": "disjoint human validation",
               "nominal_fpr": .01, "threshold": threshold, "conditions": {}}
    if stage == "text_base":
        base, _ = load_model(manifest["qwen"]["path"])
        clean_state = {name: p.detach().cpu() for name, p in base.named_parameters()}
        del base
        gc.collect(); torch.cuda.empty_cache()
        watermarked = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        variants = [(f"watermarked_weight_{weight:g}", weight) for weight in (.75, .5, .25, 0.)]
    else:
        variants = [("watermarked", None)]
        if stage == "textseal":
            variants += [("ordinary_sampler_without_original_weights", None)]
    for name, weight in variants:
        if weight is not None:
            with torch.no_grad():
                for key, p in model.named_parameters():
                    p.copy_(weight * watermarked[key].to(p.device) + (1-weight) * clean_state[key].to(p.device))
        sampled = generate(model, tokenizer, data["test"], output / f"{name}.json", generation, stop,
                           wm=wm if stage == "textseal" and name == "watermarked" else None)
        scores, records = score_texts(tokenizer, [r["text"] for r in sampled], wm, proxy)
        metrics = detection_metrics(scores, null_test, threshold)
        metrics.update(quality(model, data["test"]))
        metrics["mean_generated_tokens"] = float(np.mean([len(r["tokens"]) for r in sampled]))
        metrics["mean_trigram_repetition"] = float(np.mean([
            1-len(set(zip(ids, ids[1:], ids[2:])))/max(len(ids)-2, 1) if len(ids) >= 3 else 0.
            for ids in [r["tokens"] for r in sampled]]))
        metrics["nominal_p001_detection"] = float(np.mean([r["p_value"] < .001 for r in records]))
        write_json(output / f"{name}_scores.json", records)
        results["conditions"][name] = metrics
        write_json(output / "results.json", results)
    if stage == "textseal":
        # The watermarked and bypassed conditions use exactly the same weights.
        results["with_original_weights"] = "same ordinary-sampler bypass; identical to no-original-weights case"
        sampled = json.loads((output / "watermarked.json").read_text())
        mixed = [human[i] + "\n" + r["text"] + "\n" + human[(i+1)%len(human)] + "\n" + human[(i+2)%len(human)]
                 for i, r in enumerate(sampled)]
        scores, _ = score_texts(tokenizer, mixed, wm, proxy)
        results["dilution_approximately_fourfold"] = {
            "nominal_p001_detection": float(np.mean(np.asarray(scores) > 3)),
            "note": "Uses the corrected localized p-value; differs in document length from the calibrated test."}
    elif stage == "distilled":
        del model
        gc.collect(); torch.cuda.empty_cache()
        base, _ = load_model(manifest["qwen"]["path"])
        for name, teacher_wm in (("initial_base", None), ("biased_teacher", wm)):
            sampled = generate(base, tokenizer, data["test"], output / f"{name}.json", generation, stop, teacher_wm)
            scores, _ = score_texts(tokenizer, [r["text"] for r in sampled], wm)
            results["conditions"][name] = {**detection_metrics(scores, null_test, threshold), **quality(base, data["test"])}
    reference = model if stage == "textseal" else base if stage == "distilled" else load_model(manifest["qwen"]["path"])[0]
    for name, metrics in results["conditions"].items():
        sampled = json.loads((output / f"{name}.json").read_text())
        metrics.update(generation_perplexity(reference, tokenizer, sampled))
    results["quality_evaluator"] = "original Qwen, evaluated after removal; never supplied to the no-original-weights optimizer"
    results["training_original_weights_access"] = stage == "text_base"
    write_json(output / "results.json", results)
