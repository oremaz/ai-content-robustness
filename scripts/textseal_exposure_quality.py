"""Freeze preservation-based API-attack selection before TextSeal2 owner scoring."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

from watermark_survival_pilot import ROOT, save, sha


ARMS = ["pre_direct", "pre_smooth", "conditional_direct", "conditional_smooth"]


def prepare(parent):
    parent = Path(parent)
    original = json.loads(parent.read_text())
    generated = Path(original["directory"])
    completion = json.loads((generated / "generation_complete.json").read_text())
    if completion["signature"] != original["signature"]:
        raise ValueError("Generation protocol changed")
    for name, expected in completion["outputs"].items():
        if sha(generated / name) != expected:
            raise ValueError("Frozen generation output changed")
    directory = ROOT / "results/research_workflow/paper_writing" / (
        "text-exposure-quality-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    source = directory / "source"
    (source / "scripts").mkdir(parents=True)
    shutil.copytree(ROOT / "ai_content_detector", source / "ai_content_detector",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.ipynb", ".pytest_cache"))
    for name in ["textseal_exposure_quality.py", "watermark_survival_pilot.py", "watermark_transport_train.py",
                 "run_textseal_exposure_quality.slurm"]:
        shutil.copyfile(ROOT / "scripts" / name, source / "scripts" / name)
    config = {**original, "directory": str(directory), "source": str(source), "generation_manifest": str(parent),
              "generated": str(generated), "compute": {"qos": "qos_gpu_h100-dev", "gpus": 1,
                                                         "minutes": 30, "maximum_allocations": 2}}
    files = [parent, generated / "generation_complete.json", Path(original["runtime_archive"]["path"])]
    files += [generated / name for name in completion["outputs"]]
    for model in [original["jina"], original["models"]["cross-encoder/nli-deberta-v3-large"],
                  original["models"]["mistralai/Mistral-7B-v0.1"]]:
        files += [p for p in Path(model["path"]).rglob("*") if p.is_file()]
    config["input_hashes"] = {str(p): sha(p) for p in files}
    config["source_hashes"] = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    config.pop("signature", None)
    config["signature"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    save(directory / "manifest.json", config)
    print(directory / "manifest.json")


def run(path):
    print("text preservation worker starting", flush=True)
    import numpy as np
    import torch
    from unittest.mock import patch
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
    from ai_content_detector.watermarking.semantic_similarity import last_token_pool, text_key
    from ai_content_detector.watermarking.text_attacks import degenerated

    config = json.loads(Path(path).read_text())
    arms = config.get("attack_arms", ARMS)
    root, generated = Path(config["directory"]), Path(config["generated"])
    for filename, expected in config["input_hashes"].items():
        if sha(filename) != expected:
            raise ValueError("Preservation input changed: " + filename)
    for relative, expected in config["source_hashes"].items():
        if sha(Path(config["source"]) / relative) != expected:
            raise ValueError("Preservation source changed: " + relative)
    with patch("importlib.metadata.packages_distributions", return_value=config["package_map"]):
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
    originals = json.loads((generated / "originals.json").read_text())
    attempts = {arm: json.loads((generated / (arm + ".json")).read_text()) for arm in arms}
    expected = [(r["id"], a) for r in originals for a in range(config["sampling"]["attempts"])]
    if any([(r["id"], r["attempt"]) for r in rows] != expected for rows in attempts.values()):
        raise ValueError("Attack attempts are not aligned with original IDs")
    original_by_id = {r["id"]: r for r in originals}
    texts = {text_key(r["text"]): r["text"] for rows in [originals] + list(attempts.values()) for r in rows}
    save(root / "environment.json", {"job_id": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__,
                                     "python": sys.version, "gpu": torch.cuda.get_device_name()})
    with StopRequest() as stop, torch.inference_mode():
        vectors = {}
        embedder_path = config["jina"]["path"]
        tokenizer = AutoTokenizer.from_pretrained(embedder_path, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(embedder_path, local_files_only=True, dtype=torch.bfloat16,
                                           attn_implementation="sdpa").cuda().eval().requires_grad_(False)
        keys = sorted(texts)
        for start in range(0, len(keys), 16):
            batch = keys[start:start+16]
            cache = root / f"embeddings-{start:05}.npz"
            if cache.exists():
                with np.load(cache, allow_pickle=False) as saved:
                    if saved["keys"].tolist() != batch:
                        raise ValueError("Embedding restart texts changed")
                    values = saved["vectors"]
            else:
                encoded = tokenizer(["Document: " + texts[k] for k in batch], padding=True, truncation=False,
                                    return_tensors="pt").to("cuda")
                if encoded.input_ids.shape[1] > 32768:
                    raise ValueError("Embedding input exceeds context")
                hidden = model(**encoded, use_cache=False).last_hidden_state
                values = torch.nn.functional.normalize(last_token_pool(hidden, encoded.attention_mask).float(), dim=-1).cpu().numpy()
                np.savez_compressed(cache, keys=np.asarray(batch), vectors=values)
                del encoded, hidden
            vectors.update(zip(batch, values))
            if stop.requested:
                raise CheckpointPause("Embedding batch saved")
        del model
        torch.cuda.empty_cache()
        ppl_path = config["models"]["mistralai/Mistral-7B-v0.1"]["path"]
        tokenizer = AutoTokenizer.from_pretrained(ppl_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(ppl_path, local_files_only=True, dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").cuda().eval().requires_grad_(False)
        cache_path = root / "perplexities.json"
        ppls = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        logits, tokens = None, None
        for key in keys:
            if key in ppls:
                continue
            tokens = tokenizer(texts[key], add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            value = None
            if tokens.shape[-1] > 1:
                logits = model(tokens, use_cache=False).logits[:, :-1].float()
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
                loss_value = float(loss)
                if math.isfinite(loss_value) and loss_value < 700:
                    value = math.exp(loss_value)
            ppls[key] = value
            save(cache_path, ppls)
            if stop.requested:
                raise CheckpointPause("Perplexity item saved")
        del model, logits, tokens
        torch.cuda.empty_cache()
        nli_path = config["models"]["cross-encoder/nli-deberta-v3-large"]["path"]
        tokenizer = AutoTokenizer.from_pretrained(nli_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(nli_path, local_files_only=True,
                                                                  dtype=torch.float32).cuda().eval().requires_grad_(False)
        label_names = {int(i): name.lower() for i, name in model.config.id2label.items()}
        if set(label_names.values()) != {"contradiction", "entailment", "neutral"}:
            raise ValueError("Unexpected NLI class mapping")
        selected = {}
        for arm, rows in attempts.items():
            output = root / (arm + "-quality.json")
            saved = json.loads(output.read_text()) if output.exists() else []
            if [(r["id"], r["attempt"]) for r in saved] != expected[:len(saved)]:
                raise ValueError("Quality restart IDs changed")
            for row in rows[len(saved):]:
                original = original_by_id[row["id"]]
                first, second = original["text"], row["text"]
                directions = [(first, second), (second, first)]
                lengths = [len(tokenizer(a, b, truncation=False).input_ids) for a, b in directions]
                labels, confidences = [], []
                if max(lengths) <= 512:
                    encoded = tokenizer([a for a, b in directions], [b for a, b in directions], padding=True,
                                        truncation=False, return_tensors="pt").to("cuda")
                    probabilities = model(**encoded).logits.softmax(-1)
                    labels = [label_names[i] for i in probabilities.argmax(-1).tolist()]
                    confidences = probabilities.max(-1).values.tolist()
                a, b = vectors[text_key(first)], vectors[text_key(second)]
                cosine = float(np.clip(np.dot(a, b), -1, 1))
                numbers = lambda text: set(re.findall(r"\b\d+(?:[.,]\d+)*\b", text))
                a_numbers, b_numbers = numbers(first), numbers(second)
                numeric_f1 = 2 * len(a_numbers & b_numbers)/(len(a_numbers)+len(b_numbers)) if a_numbers or b_numbers else 1.
                original_ppl, candidate_ppl = ppls[text_key(first)], ppls[text_key(second)]
                ratio = candidate_ppl/original_ppl if original_ppl and candidate_ppl else None
                failures = []
                if not first.strip() or not second.strip():failures.append("empty")
                if row["length_capped"] or original["length_capped"]:failures.append("length_cap")
                if max(lengths)>512:failures.append("nli_truncated")
                if labels != ["entailment", "entailment"]:failures.append("bidirectional_nli")
                if not math.isfinite(cosine) or cosine < config["quality_rule"]["jina_min"]:failures.append("jina")
                if numeric_f1 != 1.:failures.append("numeric_set")
                if ratio is None or not math.isfinite(ratio) or not 0 < ratio <= config["quality_rule"]["max_ppl_ratio"]:failures.append("perplexity")
                if degenerated(row["tokens"]) or re.search(r"[A-Za-z]{60,}", second):failures.append("repetition_or_long_letter_run")
                if any(marker in second for marker in ["<document>", "</document>"]):failures.append("wrapper_leakage")
                saved.append({"id": row["id"], "attempt": row["attempt"], "jina_cosine": cosine,
                              "nli_labels": labels, "nli_confidences": confidences, "nli_lengths": lengths,
                              "numeric_set_f1": numeric_f1, "original_ppl": original_ppl, "candidate_ppl": candidate_ppl,
                              "perplexity_ratio": ratio, "passed": not failures, "failures": failures,
                              "original_sha256": text_key(first), "candidate_sha256": text_key(second)})
                save(output, saved)
                if stop.requested:
                    raise CheckpointPause("Preservation item saved")
            selected[arm] = []
            for original in originals:
                eligible = [r for r in saved if r["id"] == original["id"]]
                choice = next((r for r in eligible if r["passed"]), eligible[0])
                selected[arm].append({"id": choice["id"], "attempt": choice["attempt"],
                                      "passed": choice["passed"], "candidate_sha256": choice["candidate_sha256"]})
        save(root / "selection.json", {"signature": config["signature"], "selected": selected,
                                       "policy": config["selection"], "owner_scores_used": False})
        save(root / "quality_complete.json", {"signature": config["signature"],
            "selection_sha256": sha(root / "selection.json"), "quality_hashes": {arm: sha(root/(arm+"-quality.json")) for arm in arms},
            "status": "Quality-based selection frozen. Owner scoring remains a separate phase."})
        print({arm: sum(r["passed"] for r in rows) for arm, rows in selected.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("manifest")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.manifest)
    else:
        from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
        try:
            run(args.manifest)
        except CheckpointPause as error:
            print(str(error), flush=True)
            sys.exit(75)
