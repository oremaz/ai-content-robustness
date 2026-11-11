"""Instruction-formatted inputs and the released TextSeal2 detector."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def chat_prompt(tokenizer, question):
    return tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def prepare_questions(path, tokenizer, seed, counts=None):
    counts = counts or {"train": 1000, "validation": 256, "test": 256, "removal": 200}
    groups = {name: [] for name in counts}
    seen = set()
    seen_answers = set()
    for line in Path(path).read_text().splitlines():
        item = json.loads(line)
        question = item["question"].strip()
        key = hashlib.sha256(question.lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        bucket = int(key[:12], 16) % 100
        split = "train" if bucket < 65 else "removal" if bucket < 85 else "validation" if bucket < 90 else "test"
        prompt = chat_prompt(tokenizer, question)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        if not prefix or len(prefix) > 768:
            continue
        for answer in item["human_answers"]:
            response = tokenizer.encode(answer, add_special_tokens=False)
            if len(response) < 256:
                continue
            response = response[:256]
            answer_key=hashlib.sha256(json.dumps(response).encode()).hexdigest()
            if answer_key in seen_answers:
                continue
            seen_answers.add(answer_key)
            groups[split].append({"id": key, "question": question, "prompt": prompt,
                "prompt_length": len(prefix), "tokens": prefix+response,
                "human_tokens": response, "human_text": tokenizer.decode(response, skip_special_tokens=False)})
            break
    rng = np.random.default_rng(seed)
    for name, number in counts.items():
        rng.shuffle(groups[name])
        if len(groups[name]) < number:
            raise ValueError(f"Insufficient independent {name} questions: {len(groups[name])} < {number}")
        groups[name] = groups[name][:number]
    return {"text": groups, "seed": seed, "format": "Full user question and native thinking-disabled chat template; response-only training."}


def textseal2_config(key=42):
    from textseal.watermarking.config import WatermarkConfig
    return WatermarkConfig(watermark_type="textseal2", secret_key=key, ngram=3, rho=.9,
                           prf="hires", repeated_windows="skip", scoring_method="v2")


def textseal2_score(detector, text):
    scores = detector.get_scores_by_t([text], scoring_method="v2", seen_windows=set())[0]
    probability = detector.get_pvalue(math.fsum(scores), len(scores))
    return {"score": -math.log10(max(probability, 1e-200)), "p_value": probability,
            "scored_pairs": len(scores), "eligible": len(scores) > 0}


def textseal2_scores(detector, texts, chunk_size=65536):
    """Batch the native v2 primitive while retaining per-document pair order."""
    import torch
    from textseal.watermarking.core import copula_scores

    if detector.wm_args.watermark_type != "textseal2" or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Expected the native TextSeal2 detector and a positive chunk size")
    width = detector.ngram
    tokens = [detector.tokenizer.encode(text, add_special_tokens=False) if isinstance(text, str) else text
              for text in texts]
    groups = [list(dict.fromkeys(tuple(ids[t-width:t+1]) for t in range(width, len(ids)))) for ids in tokens]
    flat = [pair for group in groups for pair in group]
    values = []
    with torch.inference_mode():
        for start in range(0, len(flat), chunk_size):
            pairs = torch.tensor(flat[start:start+chunk_size], dtype=torch.long)
            values.extend(copula_scores(pairs[:, :-1], pairs[:, -1],
                [detector.wm_args.key_a, detector.wm_args.key_b], detector.wm_args.prf).tolist())
    result, start = [], 0
    for group in groups:
        total = math.fsum(values[start:start+len(group)])
        start += len(group)
        probability = detector.get_pvalue(total, len(group))
        result.append({"score": -math.log10(max(probability, 1.e-200)), "p_value": probability,
                       "scored_pairs": len(group), "eligible": bool(group), "evidence_sum": total})
    return result


def sampling_settings():
    return {"batch": 8, "max_tokens": 200, "temperature": 1., "top_p": .95,
            "top_k": 0, "thinking": False}
