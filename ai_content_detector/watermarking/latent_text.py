"""Experimental DistSeal-inspired latent teacher and text-only message extractor."""

from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..rl_evasion.checkpointing import (
    CheckpointClock, CheckpointPause, atomic_save, rng_state, restore_rng,
)
from .protocol import detection_metrics, operating_point, write_json
from .text import fit, forward_kl, generate, generation_perplexity, load_model, quality


class LatentEmbedder(nn.Module):
    def __init__(self, hidden, bits, width=128, strength=.35):
        super().__init__()
        self.down = nn.Linear(hidden, width)
        self.message = nn.Linear(bits, width, bias=False)
        self.up = nn.Linear(width, hidden)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.strength = strength

    def forward(self, hidden, message):
        dtype = hidden.dtype
        hidden = hidden.float()
        rms = hidden.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        message = message.float()
        if message.ndim == 1:
            message = message[None].expand(hidden.shape[0], -1)
        code = self.message(message * 2 - 1)[:, None]
        residual = self.up(F.silu(self.down(hidden / rms) + code)).tanh()
        return (hidden + self.strength * rms * residual).to(dtype)


class TextExtractor(nn.Module):
    def __init__(self, vocab, bits, width=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.convolutions = nn.ModuleList([nn.Conv1d(width, width, 3, padding=1) for _ in range(2)])
        self.output = nn.Linear(width, bits)

    def forward(self, ids=None, mask=None, embeddings=None):
        x = self.embedding(ids) if embeddings is None else embeddings
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if mask is None else mask
        keep = mask[:, None].to(x.dtype)
        x = x.transpose(1, 2) * keep
        for layer in self.convolutions:
            x = (x + F.gelu(layer(x))) * keep
        pooled = x.sum(-1) / keep.sum(-1).clamp_min(1)
        return self.output(pooled)


def pack_tokens(sequences, device, max_length=1024):
    sequences = [list(ids[:max_length]) for ids in sequences]
    size = max(1, max(map(len, sequences)))
    ids = torch.zeros((len(sequences), size), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(sequences):
        ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[i, :len(row)] = True
    return ids, mask


def text_tokens(tokenizer, sampled):
    return [tokenizer.encode(tokenizer.decode(ids.tolist(), skip_special_tokens=True),
                             add_special_tokens=False) for ids in sampled.detach().cpu()]


def straight_through_logits(extractor, hard_ids, hard_mask, soft_embeddings, soft_mask):
    hard = extractor(hard_ids, hard_mask)
    state = {name: value.detach() for name, value in extractor.named_parameters()}
    soft = torch.func.functional_call(extractor, state, (),
                                     {"embeddings": soft_embeddings, "mask": soft_mask})
    # The forward detector sees decoded, retokenized text. Only the embedder
    # gradient uses a continuous surrogate; extractor training uses hard text.
    return hard + (soft - soft.detach())


def channel(embedder, extractor, head, hidden, messages, temperature=.8, chunk=32):
    batch, length, width = hidden.shape
    watermarked = embedder(hidden, messages)
    soft, sampled, divergences = [], [], []
    base_hidden, marked_hidden = hidden.reshape(-1, width), watermarked.reshape(-1, width)
    for start in range(0, len(base_hidden), chunk):
        with torch.no_grad():
            baseline = head(base_hidden[start:start+chunk]).float().log_softmax(-1)
        logits = head(marked_hidden[start:start+chunk]).float()
        logq = logits.log_softmax(-1)
        divergences.append((logq.exp() * (logq - baseline)).sum(-1))
        q = (logits / temperature).softmax(-1)
        sampled.append(torch.multinomial(q.detach(), 1).squeeze(-1))
        soft.append(q @ extractor.embedding.weight.detach().to(q.dtype))
    return (torch.cat(sampled).reshape(batch, length),
            torch.cat(soft).reshape(batch, length, -1), torch.cat(divergences).mean())


def train_batch(model, tokenizer, embedder, extractor, rows, config, generator):
    length = config["length"]
    ids = torch.tensor([row["tokens"][:length+50] for row in rows], device="cuda")
    with torch.no_grad():
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, 49:49+length]
    hidden = hidden.repeat_interleave(2, 0)
    message = torch.randint(2, (len(rows), config["bits"]), generator=generator, device="cuda").float()
    messages = torch.stack((message, 1-message), 1).flatten(0, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sampled, soft, divergence = channel(embedder, extractor, model.get_output_embeddings(),
                                            hidden, messages, config["temperature"])
    hard_ids, hard_mask = pack_tokens(text_tokens(tokenizer, sampled), "cuda")
    soft_mask = torch.ones(soft.shape[:2], device="cuda", dtype=torch.bool)
    logits = straight_through_logits(extractor, hard_ids, hard_mask, soft.float(), soft_mask)
    bce = F.binary_cross_entropy_with_logits(logits, messages)
    accuracy = ((logits.detach() >= 0) == messages.bool()).float().mean()
    return bce, divergence, accuracy


def architecture(model, config):
    vocab, hidden = model.get_output_embeddings().weight.shape
    return {"hidden": hidden, "vocab": vocab, "bits": config["bits"],
            "width": config["width"], "strength": config["strength"]}


def make_pair(spec, device="cuda"):
    embedder = LatentEmbedder(spec["hidden"], spec["bits"], spec["width"], spec["strength"]).to(device)
    extractor = TextExtractor(spec["vocab"], spec["bits"], spec["width"]).to(device)
    return embedder, extractor


def load_pair(manifest, device="cuda"):
    path = Path(manifest["directory"]) / "latent_embedder/selected.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["protocol"] != manifest["protocol_sha256"]:
        raise ValueError("Latent watermark protocol mismatch")
    embedder, extractor = make_pair(state["architecture"], device)
    embedder.load_state_dict(state["embedder"])
    extractor.load_state_dict(state["extractor"])
    return embedder.eval().requires_grad_(False), extractor.eval().requires_grad_(False), state


def install_embedder(model, embedder, message):
    message = torch.tensor(message, dtype=torch.float32, device=next(embedder.parameters()).device)
    def hook(layer, args):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args[0].is_cuda):
            return (embedder(args[0], message),) + args[1:]
    return model.get_output_embeddings().register_forward_pre_hook(hook)


@contextmanager
def embedded(model, embedder, message):
    handle = install_embedder(model, embedder, message)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def score(extractor, tokenizer, texts, message):
    records = []
    target = torch.tensor(message, device=next(extractor.parameters()).device).bool()
    for start in range(0, len(texts), 32):
        ids, mask = pack_tokens([tokenizer.encode(t, add_special_tokens=False) for t in texts[start:start+32]],
                                target.device)
        logits = extractor(ids, mask)
        values = F.logsigmoid(logits * (target.float()*2-1)).mean(-1)
        accuracy = ((logits >= 0) == target).float().mean(-1)
        records.extend({"score": float(s) if n >= 32 else -1e6, "bit_accuracy": float(a), "tokens": int(n),
                        "eligible": bool(n >= 32)}
                       for s, a, n in zip(values, accuracy, mask.sum(-1)))
    return [r["score"] for r in records], records


@torch.no_grad()
def validate(model, tokenizer, embedder, extractor, rows, config, seed):
    saved = rng_state()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    torch.manual_seed(seed)
    values = []
    embedder.eval(); extractor.eval()
    for start in range(0, len(rows), config["batch"]):
        bce, kl, accuracy = train_batch(model, tokenizer, embedder, extractor,
                                       rows[start:start+config["batch"]], config, generator)
        values.append([float(bce), float(kl), float(accuracy)])
    embedder.train(); extractor.train()
    restore_rng(saved)
    mean = np.mean(values, axis=0)
    return {"bce": float(mean[0]), "kl_nats_per_token": float(mean[1]), "bit_accuracy": float(mean[2])}


def train(manifest, stop):
    output = Path(manifest["directory"]) / "latent_embedder"
    output.mkdir(exist_ok=True)
    config = manifest["latent_training"]
    data = json.loads(Path(manifest["data"]).read_text())["text"]
    model, tokenizer = load_model(manifest["qwen"]["path"])
    spec = architecture(model, config)
    embedder, extractor = make_pair(spec)
    params = list(embedder.parameters()) + list(extractor.parameters())
    optimizer = torch.optim.AdamW(params, lr=config["lr"], weight_decay=.01)
    generator = torch.Generator(device="cuda").manual_seed(manifest["seed"])
    step, beta, best, history = 0, 1., -1., []
    restart = output / "restart.pt"
    if restart.exists():
        state = torch.load(restart, map_location="cpu", weights_only=False)
        if state["protocol"] != manifest["protocol_sha256"] or state["architecture"] != spec:
            raise ValueError("Latent training restart mismatch")
        embedder.load_state_dict(state["embedder"]); extractor.load_state_dict(state["extractor"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], generator)
        step, beta, best, history = state["step"], state["beta"], state["best"], state["history"]
        del state
    clock = CheckpointClock(600)
    def state_dict():
        return {"protocol": manifest["protocol_sha256"], "architecture": spec, "step": step,
                "embedder": embedder.state_dict(), "extractor": extractor.state_dict()}
    def save():
        atomic_save({**state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(generator),
                     "beta": beta, "best": best, "history": history}, restart)
        write_json(output / "progress.json", {"step": step, "beta": beta, "best": best, "history": history})
        clock.saved()
    while step < config["steps"]:
        indices = np.random.default_rng(manifest["seed"]+step).choice(len(data["train"]), config["batch"], replace=False)
        optimizer.zero_grad(set_to_none=True)
        bce, kl, accuracy = train_batch(model, tokenizer, embedder, extractor,
                                       [data["train"][i] for i in indices], config, generator)
        loss = bce + beta * kl
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite latent watermark loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite latent watermark gradient")
        optimizer.step()
        beta = float(np.clip(beta + .05*(float(kl.detach())-config["kl_budget"]), .05, 20))
        step += 1
        if step % 50 == 0:
            row = {"step": step, "loss": float(loss.detach()), "bce": float(bce.detach()),
                   "kl_nats_per_token": float(kl.detach()), "bit_accuracy": float(accuracy), "beta": beta}
            history.append(row)
            print("latent_embedder", row, flush=True)
        if step % config["validate_every"] == 0 or step == config["steps"]:
            validation = validate(model, tokenizer, embedder, extractor, data["validation"][:32], config,
                                  manifest["seed"]+100000)
            print("latent_validation", step, validation, flush=True)
            history.append({"step": step, "validation": validation})
            if validation["kl_nats_per_token"] <= config["kl_budget"]*1.25 and validation["bit_accuracy"] > best:
                best = validation["bit_accuracy"]
                atomic_save({**state_dict(), "validation": validation}, output / "selected.pt")
        if step % config["validate_every"] == 0 or clock.due() or stop.requested or step == config["steps"]:
            save()
        if stop.requested:
            raise CheckpointPause("Latent embedder and extractor saved")
    if best < config["minimum_bit_accuracy"]:
        raise RuntimeError(f"Hard-text validation failed: best bit accuracy {best:.3f}; student training blocked")
    write_json(output / "results.json", {"steps": step, "best_validation_bit_accuracy": best,
               "selection": "hard decoded/retokenized validation text under KL budget", "history": history})


def generation_config(manifest):
    return {"batch": 8, "seed": manifest["seed"], "top_p": 1.,
            "temperature": manifest["latent_training"]["temperature"]}


def teacher_probe(manifest, stop):
    output = Path(manifest["directory"]) / "latent_teacher"
    output.mkdir(exist_ok=True)
    data = json.loads(Path(manifest["data"]).read_text())["text"]
    model, tokenizer = load_model(manifest["qwen"]["path"])
    embedder, extractor, selected = load_pair(manifest)
    message = manifest["latent_training"]["message"]
    rows = data["validation"][:64]
    base = generate(model, tokenizer, rows, output / "base.json", generation_config(manifest), stop)
    with embedded(model, embedder, message):
        marked = generate(model, tokenizer, rows, output / "teacher.json", generation_config(manifest), stop)
    negative, base_records = score(extractor, tokenizer, [r["text"] for r in base], message)
    positive, records = score(extractor, tokenizer, [r["text"] for r in marked], message)
    threshold = operating_point(negative, .05)
    metrics = detection_metrics(positive, negative, threshold)
    metrics.update({"bit_accuracy": float(np.mean([r["bit_accuracy"] for r in records])),
                    "base_bit_accuracy": float(np.mean([r["bit_accuracy"] for r in base_records])),
                    "selected_step": selected["step"], "selection_data": "validation only",
                    "native_kl_nats_per_token": selected["validation"]["kl_nats_per_token"]})
    metrics["passed"] = (metrics["bit_accuracy"] >= manifest["latent_training"]["minimum_bit_accuracy"]
                         and metrics["auroc"] >= .7 and metrics["detection"]["rate"] >= .25)
    write_json(output / "results.json", metrics)
    if not metrics["passed"]:
        raise RuntimeError("Autoregressive teacher watermark failed validation; long student jobs blocked")


def preflight(manifest):
    output = Path(manifest["directory"]) / "latent_preflight"
    output.mkdir(exist_ok=True)
    model, tokenizer = load_model(manifest["qwen"]["path"])
    config = manifest["latent_training"]
    embedder, extractor = make_pair(architecture(model, config))
    row = json.loads(Path(manifest["data"]).read_text())["text"]["train"][:1]
    generator = torch.Generator(device="cuda").manual_seed(manifest["seed"])
    bce, kl, accuracy = train_batch(model, tokenizer, embedder, extractor, row,
                                    {**config, "length": 16}, generator)
    (bce+kl).backward()
    gradient = torch.nn.utils.clip_grad_norm_(list(embedder.parameters())+list(extractor.parameters()), 1.)
    if not torch.isfinite(gradient) or embedder.up.weight.grad.abs().sum() == 0:
        raise RuntimeError("Latent straight-through gradient failed")
    embedder.eval().requires_grad_(False)
    with torch.no_grad(), embedded(model, embedder, config["message"]):
        generated = generate(model, tokenizer, row, output / "generated.json",
                             {**generation_config(manifest), "max_tokens": 8}, type("Stop", (), {"requested": False})())
    _, records = score(extractor.eval(), tokenizer, [r["text"] for r in generated], config["message"])
    # Validate the exact dense KL and LoRA path used by the merged student.
    del model, embedder, extractor
    gc.collect(); torch.cuda.empty_cache()
    student, _ = load_model(manifest["qwen"]["path"], training=True)
    ids = torch.tensor([row[0]["tokens"][:8]], device="cuda")
    logits = student(ids, use_cache=False).logits[:, :-1]
    target = logits.detach().roll(1, -1)
    distill_loss = forward_kl(logits, target, torch.ones_like(ids[:, :-1], dtype=torch.bool))
    distill_loss.backward()
    student_grad = torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.)
    if not torch.isfinite(student_grad) or student_grad <= 0:
        raise RuntimeError("Latent student backward failed")
    write_json(output / "results.json", {"bce": float(bce.detach()), "kl": float(kl.detach()),
               "hard_bit_accuracy": float(accuracy), "embedder_extractor_gradient_norm": float(gradient),
               "student_gradient_norm": float(student_grad), "generated": records})


def summarize(extractor, tokenizer, sampled, message, human, base, thresholds):
    scores, records = score(extractor, tokenizer, [r["text"] for r in sampled], message)
    h0, _ = score(extractor, tokenizer, human, message)
    b0, _ = score(extractor, tokenizer, [r["text"] for r in base], message)
    metrics = detection_metrics(scores, h0, thresholds["combined"])
    metrics["against_base"] = detection_metrics(scores, b0, thresholds["combined"])
    metrics["bit_accuracy"] = float(np.mean([r["bit_accuracy"] for r in records]))
    metrics["mean_generated_tokens"] = float(np.mean([len(r["tokens"]) for r in sampled]))
    metrics["mean_trigram_repetition"] = float(np.mean([
        1-len(set(zip(t, t[1:], t[2:])))/max(len(t)-2, 1) if len(t) >= 3 else 0.
        for t in [r["tokens"] for r in sampled]]))
    return metrics, records


def keyless_probe(base_train, marked_train, base_test, marked_test, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    train_text = [r["text"] for r in base_train+marked_train]
    test_text = [r["text"] for r in base_test+marked_test]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=30000)
    features = vectorizer.fit_transform(train_text)
    classifier = LogisticRegression(C=1., max_iter=500, random_state=seed)
    classifier.fit(features, [0]*len(base_train)+[1]*len(marked_train))
    probabilities = classifier.predict_proba(vectorizer.transform(test_text))[:, 1]
    return {"auroc": float(roc_auc_score([0]*len(base_test)+[1]*len(marked_test), probabilities)),
            "training_documents": len(train_text), "test_documents": len(test_text),
            "extractor_or_message_access": False,
            "interpretation": "bounded trained text distinguisher, not a security proof"}


def evaluate(manifest, stage, stop):
    root = Path(manifest["directory"])
    output = root / stage
    output.mkdir(exist_ok=True)
    data = json.loads(Path(manifest["data"]).read_text())["text"]
    message = manifest["latent_training"]["message"]
    embedder, extractor, selected = load_pair(manifest)
    config = generation_config(manifest)
    if stage == "latent_clean":
        fit(manifest, stop, clean=True)
    reference, tokenizer = load_model(manifest["qwen"]["path"])
    controls = root / "latent_evaluate"
    controls.mkdir(exist_ok=True)
    base = generate(reference, tokenizer, data["test"], controls / "base.json", config, stop)
    human = [tokenizer.decode(r["tokens"][50:306]) for r in data["test"]]
    calibration_path = controls / "calibration.json"
    if calibration_path.exists():
        thresholds = json.loads(calibration_path.read_text())
    else:
        calibration = generate(reference, tokenizer, data["validation"], controls / "base_validation.json", config, stop)
        human_cal, _ = score(extractor, tokenizer, [tokenizer.decode(r["tokens"][50:306]) for r in data["validation"]], message)
        base_cal, _ = score(extractor, tokenizer, [r["text"] for r in calibration], message)
        thresholds = {"combined": max(operating_point(human_cal), operating_point(base_cal)),
                      "nominal_fpr": .01, "human_n": len(human_cal), "base_n": len(base_cal),
                      "calibration": "max of disjoint human and base validation 99th percentiles"}
        write_json(calibration_path, thresholds)
    results = {"method": "distseal_latent_text", "selected_teacher_step": selected["step"],
               "thresholds": thresholds, "conditions": {}, "ordinary_sampler": True,
               "attack_original_weights_access": stage == "latent_base"}
    def record(name, model, sampled):
        metrics, records = summarize(extractor, tokenizer, sampled, message, human, base, thresholds)
        metrics.update(quality(model, data["test"]))
        metrics.update(generation_perplexity(reference, tokenizer, sampled))
        results["conditions"][name] = metrics
        write_json(output / f"{name}_scores.json", records)
        write_json(output / "results.json", results)
    if stage == "latent_evaluate":
        record("base", reference, base)
        with embedded(reference, embedder, message):
            teacher = generate(reference, tokenizer, data["test"], output / "teacher.json", config, stop)
        # Native model quality is measured after removing the teacher hook.
        record("external_latent_teacher", reference, teacher)
    source = root / (manifest["clean_stage"]+"/model" if stage == "latent_clean" else manifest["distilled_model"])
    student, _ = load_model(source)
    if stage == "latent_base":
        base_state = {n: p.detach().cpu().clone() for n, p in reference.named_parameters()}
        marked_state = {n: p.detach().cpu().clone() for n, p in student.named_parameters()}
        for weight in (.75, .5, .25, 0.):
            with torch.no_grad():
                for name, p in student.named_parameters():
                    p.copy_((weight * marked_state[name] + (1-weight)*base_state[name]).to(p.device))
            name = f"watermarked_weight_{weight:g}"
            sampled = generate(student, tokenizer, data["test"], output / f"{name}.json", config, stop)
            record(name, student, sampled)
    else:
        sampled = generate(student, tokenizer, data["test"], output / "student.json", config, stop)
        record("student", student, sampled)
        wrong = np.random.default_rng(manifest["seed"]+300000).integers(2, size=(32, len(message)))
        wrong_scores = [score(extractor, tokenizer, [r["text"] for r in sampled], m)[0] for m in wrong]
        results["wrong_message_control"] = {
            "mean_detection_at_owner_threshold": float((np.asarray(wrong_scores) > thresholds["combined"]).mean()),
            "random_messages": len(wrong), "cryptographic_key_test": False}
        if stage == "latent_evaluate":
            train_base = generate(reference, tokenizer, data["train"][:128], output / "probe_base_train.json", config, stop)
            train_marked = generate(student, tokenizer, data["train"][:128], output / "probe_marked_train.json", config, stop)
            results["keyless_text_probe"] = keyless_probe(train_base, train_marked, base, sampled, manifest["seed"])
        results["random_token_deletion"] = {}
        for fraction in (.1, .3):
            def delete(rows):
                rng = np.random.default_rng(manifest["seed"]+400000)
                altered = []
                for row in rows:
                    ids = np.asarray(row["tokens"])
                    kept = ids[rng.random(len(ids)) >= fraction].tolist()
                    altered.append({**row, "text": tokenizer.decode(kept), "tokens": kept})
                return altered
            altered = delete(sampled)
            human_rows = [{"tokens": tokenizer.encode(t, add_special_tokens=False)} for t in human]
            altered_human = [r["text"] for r in delete(human_rows)]
            metrics, _ = summarize(extractor, tokenizer, altered, message, altered_human, delete(base), thresholds)
            metrics["threshold_policy"] = "frozen clean owner threshold; deletion applied to both positive and negative text"
            results["random_token_deletion"][str(fraction)] = metrics
    results["quality_evaluator"] = "original Qwen after attack; not given to clean-finetuning optimizer"
    write_json(output / "results.json", results)
