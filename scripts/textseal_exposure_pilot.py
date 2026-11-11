"""Development study of copula-consistent probability exposure and blind API rewriting."""

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

from watermark_survival_pilot import ROOT, dependency_map, save, sha


ARMS = ["pre_direct", "pre_smooth", "conditional_direct", "conditional_smooth"]


def prepare(reuse_parent=None, block_source_ngrams=False, ratio_attack=False, confirmation_shard=None, audit=False, audit_attack_prefixes=False, support_guard=False):
    parent_path = ROOT / "results/research_workflow/text_instruct/20261103T231534057688Z/manifest.json"
    prior = json.loads(parent_path.read_text())
    old = json.loads(Path(prior["data"]).read_text())["text"]
    excluded = {r["id"] for rows in old.values() for r in rows}
    if confirmation_shard is not None:
        if reuse_parent or confirmation_shard not in (0, 1):
            raise ValueError("Confirmation uses two fresh shards without reused originals")
        development = json.loads((ROOT/"results/research_workflow/paper_writing/text-exposure-blocked-20261107T183333Z/manifest.json").read_text())
        excluded |= {r["id"] for rows in development["data"].values() for r in rows}
    pools = {"calibration": [], "evaluation": []}
    seen = set(excluded)
    data = ROOT / "data/hc3-all.jsonl"
    for line in data.read_text().splitlines():
        row = json.loads(line)
        question = row["question"].strip()
        identity = hashlib.sha256(question.lower().encode()).hexdigest()
        if identity in seen or not 30 <= len(question) <= 1500 or not row["human_answers"]:
            continue
        seen.add(identity)
        bucket = int(identity[:12], 16) % 100
        split = "evaluation" if bucket >= 90 else "calibration" if bucket < 65 else None
        if split:
            pools[split].append({"id": identity, "question": question,
                                 "human_text": " ".join(row["human_answers"][0].split()[:100])})
    split_seed = "2026110711:" if confirmation_shard is not None else "2026110603:"
    for name, number in [("calibration", 64), ("evaluation", 128 if confirmation_shard is not None else 32)]:
        pools[name].sort(key=lambda r: hashlib.sha256((split_seed + r["id"]).encode()).hexdigest())
        pools[name] = pools[name][:number]
        if len(pools[name]) != number:
            raise ValueError("Insufficient fresh questions")
    if confirmation_shard is not None:
        pools["evaluation"] = pools["evaluation"][64*confirmation_shard:64*(confirmation_shard+1)]
    directory = ROOT / "results/research_workflow/paper_writing" / (
        ("text-exposure-guard-" if support_guard else "text-exposure-audit-" if audit else f"text-exposure-confirm-{confirmation_shard}-" if confirmation_shard is not None else "text-exposure-ratio-" if ratio_attack else "text-exposure-blocked-" if block_source_ngrams else "text-exposure-")
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    source = directory / "source"
    (source / "scripts").mkdir(parents=True)
    shutil.copytree(ROOT / "ai_content_detector", source / "ai_content_detector",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.ipynb", ".pytest_cache"))
    for name in ["textseal_exposure_pilot.py", "watermark_survival_pilot.py", "watermark_transport_train.py",
                 "run_textseal_exposure_pilot.slurm"]:
        shutil.copyfile(ROOT / "scripts" / name, source / "scripts" / name)
    runtime = json.loads((ROOT / "results/research_workflow/paper_writing/resources/native_python_packages.json").read_text())
    config = {"directory": str(directory), "source": str(source), "parent": str(parent_path),
              "textseal2_source": prior["textseal2_source"], "textseal2_revision": prior["textseal2_revision"],
              "generator": prior["generator"], "reference": prior["reference"], "models": prior["models"],
              "jina": json.loads((ROOT / "results/research_workflow/watermarking/resources/jina_v5_model.json").read_text()),
              "watermark": prior["watermark"], "data": pools, "seed": 2026110603,
              "runtime_archive": runtime, "package_map": dependency_map(),
              "sampling": {"temperature": 1., "top_p": .95, "max_tokens": 384, "k": 10,
                           "integration_draws": 256, "attempts": 3, "alpha": 1.},
              "original_instruction": "Answer the question in one self-contained paragraph of 80 to 100 words. State the main explanation directly. Do not use a list or heading.",
              "rewrite_instruction": "Rewrite the document below in one paragraph, preserving every factual claim, named entity and number. Change phrasing without adding or omitting information. Return only the rewritten document.",
              "quality_rule": {"jina_min": .9, "bidirectional_nli": "entailment", "numeric_set_f1": 1.,
                               "max_ppl_ratio": 1.5, "nonempty": True, "no_truncation_or_length_cap": True,
                               "no_wrapper_leakage_or_degenerated_repetition": True},
              "selection": "Select the first quality-passing attempt per prompt and arm, before owner scoring. If none passes, retain the first attempt as a failed-preservation output. Keep all attempts.",
              "scope": "New probability-interface defense; native TextSeal2 generation and detector unchanged. Arbitrary-prefix blind rewriting attacker uses only top-10 responses and an independent reference model; no key or owner scores. Development data exclude all 1712 retained questions. No first-ever or submission-readiness claim.",
              "compute": {"qos": "qos_gpu_h100-dev", "gpus": 1, "minutes": 30, "maximum_allocations": 4},
              "input_hashes": {str(parent_path): sha(parent_path), str(data): sha(data),
                               runtime["path"]: runtime["sha256"], str(Path(prior["data"])): sha(prior["data"])}}
    config["sampling"]["source_ngram_order"] = 4 if block_source_ngrams else None
    config["attack_arms"] = ["pre_ratio", "conditional_ratio"] if ratio_attack else ARMS
    if ratio_attack:
        config["sampling"]["ratio_exponents"] = [1., 2., 4.]
        config["ratio_attack"] = ("Blind exposed-top10 ratio tilt q^(1-beta)*p_reference^beta with beta1,2,4, "
            "one attempt each; identical source blocking and unchanged quality-only first-passing selection. "
            "The surrogate supplies probabilities at the exposed token IDs. No secret key or clean target probabilities. "
            "This is an interface-restricted adaptation of ratio attacks, not the published repeated-query estimator "
            "and not a guarantee of an optimal attack. Zero-mass lanes excluded; log probabilities floored at1e-12.")
    if confirmation_shard is not None:
        if not block_source_ngrams:
            raise ValueError("Confirmation requires the stronger source-blocked attack")
        config["attack_arms"] = ARMS+["pre_ratio", "conditional_ratio"]
        config["sampling"]["ratio_exponents"] = [1., 2., 4.]
        config["seed"] = 2026110711+10000000*confirmation_shard
        config["compute"]["maximum_allocations"] = 8
        config["confirmation"] = {"shard": confirmation_shard, "shards": 2, "total_questions": 128,
            "excluded_development_questions": len(excluded),
            "scope": "Fresh prompts excluding retained1712 and all96initialdevelopment/calibration identities. Same frozen interface and quality rules, one existing key, two64-prompt shards. This confirms a restricted client comparison, not multi-key or cross-dataset robustness. Native clipping versus the ideal continuous conditional estimator remains explicitly audited separately."}
    if support_guard:
        if not reuse_parent or not block_source_ngrams or ratio_attack or audit or confirmation_shard is not None:
            raise ValueError("Support guard requires its own source-blocked development pilot with reused sources")
        config["attack_arms"] = [name+"_"+attack for name in ["guard10","guard20"] for attack in ["direct","smooth","ratio"]]+["reference_only"]
        config["sampling"]["ratio_exponents"] = [1.,2.,4.]
        config["exposure_guard"] = {"coverage_loss": {"guard10": .1,"guard20": .2},
            "scope": "Provider selects a SUBSET of ordinarytop10positive-probability tokens to maximize minimum native token evidence, "
            "retaining at least(1-delta) of their estimatedconditionalmass. This changes the advertised alternatives and can return fewer than10. "
            "Reported retained probability estimates "
            "are unchanged and unnormalized. Native generation/detection unchanged. Guarantees only concern clients "
            "restricted to the advertised support; smoothing or independent reference generation can leave it. "
            "Calibration uses the same64publictexts under the new guarded interface. No owner feedback in attacker selection. "
            "This32-sourcepilotwasdesignedafterconditional-ratioattackremovedall32watermarks;freshconfirmationrequired."}
        config["compute"]["maximum_allocations"] = 8
    if audit:
        config["worker_action"] = "audit"
        config["audit"] = {"prefix_indices": list(range(0, 64, 4)), "reference_draws": 65536,
            "reference_chunk": 256, "draw_counts": [64, 256, 1024], "repeats": 3,
            "scope": "Finite-budget estimates compared with independent empirical race frequencies at16fixedpubliccalibrationprefixes. Reference frequencies are noisy, not exact probabilities. Native key/CDF clipping epsilon1e-7 versus ideal continuous construction, coupled on the same fresh noise. Report observed TV/max-coordinate error, normalizing sums, latency and omitted top10 mass."}
        config["compute"]["maximum_allocations"] = 2
    if block_source_ngrams:
        config["rewrite_instruction"] = ("Rewrite the document using substantially different wording and sentence structure. "
            "Preserve every factual claim, named entity and number without adding or omitting information. "
            "Return only one paragraph containing the rewritten document.")
        config["source_blocking"] = ("The blind client masks next-token choices that complete a source four-token sequence, "
            "after the native confidence-dependent mixture. It uses original text and public context length, never the key. "
            "If all exposed probability mass is blocked, fall back to the unmasked response and record that event. "
            "Report source-pair overlap and actual baseline evasion; copying is not a robustness result.")
    if reuse_parent:
        previous_path = Path(reuse_parent)
        previous = json.loads(previous_path.read_text())
        previous_directory = Path(previous["directory"])
        for key in ["generator", "reference", "watermark", "data", "seed"]:
            config[key] = previous[key]
        reused_files = ["originals.json"] if block_source_ngrams and not ratio_attack else ["originals.json", "confidence.json"]
        for name in reused_files:
            origin = previous_directory / name
            config["input_hashes"][str(origin)] = sha(origin)
            shutil.copyfile(origin, directory / name)
        config["reuse_parent"] = str(previous_path)
        config["input_hashes"][str(previous_path)] = sha(previous_path)
    if audit_attack_prefixes:
        if not audit or not reuse_parent:
            raise ValueError("Attack-prefix auditing requires an audit and a frozen generation parent")
        originals = json.loads((previous_directory/"originals.json").read_text())
        config["audit"]["attack_prefixes"] = [{"source_position": i, "source_id": originals[i]["id"],
            "arm": arm, "attempt": 0, "requested_position": position}
            for arm in ["pre_direct", "conditional_smooth"] for i in [0,8,16,24] for position in [0,8,32,64]]
        config["audit"]["prefix_indices"] = list(range(len(config["audit"]["attack_prefixes"])))
        config["audit"]["scope"] = ("Follow-up audit using32fixedprefixes from saved first-attempt rewrites at sourcepositions0,8,16,24 "
            "and requestedgeneratedpositions0,8,32,64 in pre_direct and conditional_smooth. Position capped at savedlength, recorded explicitly. "
            "No quality or detector filtering. Repeated generated context uses unkeyed law as in the native interface. "
            "65536freshnative/ideal reference races perprefix, noisy reference; CUDA estimator timing excludes serving overhead. "
            "This follow-up was designed after the first calibration audit showed supports1to29; it is a development audit.")
        for arm in ["pre_direct", "conditional_smooth"]:
            origin = previous_directory/(arm+".json")
            config["input_hashes"][str(origin)] = sha(origin)
    config["source_hashes"] = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    config["signature"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    save(directory / "manifest.json", config)
    print(directory / "manifest.json")


def prompt(tokenizer, instruction, document):
    return tokenizer.apply_chat_template([{"role": "user", "content": instruction + "\n\n" + document}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def run(path, audit=False):
    print("text exposure worker starting", flush=True)
    import torch
    from unittest.mock import patch
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause, StopRequest
    from ai_content_detector.rl_evasion.config import seed_everything
    from ai_content_detector.watermarking.copula_exposure import conditional_race_probabilities
    from ai_content_detector.watermarking.exposure_support import maximin_support
    from ai_content_detector.watermarking.instruct_text import textseal2_config, textseal2_score
    from ai_content_detector.watermarking.text_attacks import (
        TopKResponse, confidence, resample, smooth, source_ngram_completions, constrained_api_choice, ratio_api_choice)

    config = json.loads(Path(path).read_text())
    root = Path(config["directory"])
    for filename, expected in config["input_hashes"].items():
        if sha(filename) != expected:
            raise ValueError("Exposure-pilot input changed: " + filename)
    for relative, expected in config["source_hashes"].items():
        if sha(Path(config["source"]) / relative) != expected:
            raise ValueError("Exposure-pilot source changed: " + relative)
    sys.path.insert(0, config["textseal2_source"])
    with patch("importlib.metadata.packages_distributions", return_value=config["package_map"]):
        from ai_content_detector.watermarking.text import load_model, configured_cache
        from textseal.watermarking.generator import build_generator
        from textseal.watermarking.core import keyed_uniform, copula_scores
        from textseal.watermarking.config import DUAL_KEY_OFFSET
    settings = config["sampling"]
    model, tokenizer = load_model(config["generator"]["path"])
    if not audit:
        reference, reference_tokenizer = load_model(config["reference"]["path"])
        if tokenizer.get_vocab() != reference_tokenizer.get_vocab() or model.config.vocab_size != reference.config.vocab_size:
            raise ValueError("Probability smoothing requires identical tokenizer vocabularies")
    save(root / "environment.json", {"job_id": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__,
                                     "python": sys.version, "gpu": torch.cuda.get_device_name()})

    def public_response(logits, context, conditional=False, integration_seed=0, guard_loss=None):
        probabilities = (logits.float() / settings["temperature"]).softmax(-1)[0]
        values, ids = probabilities.sort(descending=True)
        active = values.cumsum(-1) - values <= settings["top_p"]
        values, ids = values[active], ids[active]
        values = values / values.sum()
        receipt = None
        if conditional:
            key = config["watermark"]["secret_key"]
            uniforms = [keyed_uniform(context, ids[None], k, "hires").clamp(1.e-15, 1-1.e-15)
                        for k in [key, key + DUAL_KEY_OFFSET]]
            gaussian = sum(torch.special.ndtri(u) for u in uniforms)[0] / math.sqrt(2)
            generator = torch.Generator(device="cuda").manual_seed(integration_seed)
            values, receipt = conditional_race_probabilities(values, gaussian, config["watermark"]["rho"],
                                                              settings["integration_draws"], generator)
            if guard_loss is not None:
                baseline=values.topk(min(settings["k"],len(values))).indices
                evidence = copula_scores(context.expand(len(baseline),-1),ids[baseline],[key,key+DUAL_KEY_OFFSET],"hires")
                selected, guard = maximin_support(values[baseline],evidence,settings["k"],guard_loss)
                selected=baseline[selected]
                return TopKResponse(ids[selected][None],values[selected][None],model.config.vocab_size),{**receipt,"guard":guard}
            values, order = values.sort(descending=True)
            ids = ids[order]
        return TopKResponse(ids[None, :settings["k"]], values[None, :settings["k"]], model.config.vocab_size), receipt

    def pause(stop):
        if stop.requested:
            raise CheckpointPause("Text-exposure sample/attempt saved")

    with StopRequest() as stop, torch.inference_mode():
        if audit:
            spec = config["audit"]
            output = root/"audit_results.json"
            prior = json.loads(output.read_text()) if output.exists() else {"signature": config["signature"], "rows": []}
            if prior["signature"] != config["signature"]:
                raise ValueError("Probability audit restart changed")
            rows = prior["rows"]
            if spec.get("attack_prefixes"):
                parent = json.loads(Path(config["reuse_parent"]).read_text())
                generation_root = Path(parent["directory"])
                original_by_id = {r["id"]:r for r in json.loads((generation_root/"originals.json").read_text())}
                audit_attempts = {arm:{(r["id"],r["attempt"]):r for r in json.loads((generation_root/(arm+".json")).read_text())}
                                  for arm in ["pre_direct", "conditional_smooth"]}
            for index in spec["prefix_indices"][len(rows):]:
                case = {}
                repeated = False
                if spec.get("attack_prefixes"):
                    case = spec["attack_prefixes"][index]
                    row = original_by_id[case["source_id"]]
                    prefix = tokenizer.encode(prompt(tokenizer,config["rewrite_instruction"],row["text"]),add_special_tokens=False)
                    attempt = audit_attempts[case["arm"]][(row["id"],case["attempt"])]
                    position = min(case["requested_position"],len(attempt["tokens"]))
                    continuation = attempt["tokens"][:position]
                    window = tuple((prefix+continuation)[-3:])
                    repeated = window in {tuple((prefix+continuation[:j])[-3:]) for j in range(position)}
                    case = {**case,"actual_position":position,"repeated_generated_window":repeated}
                else:
                    row = config["data"]["calibration"][index]
                    prefix = tokenizer.encode(prompt(tokenizer, config["rewrite_instruction"], row["human_text"]), add_special_tokens=False)
                    continuation = tokenizer.encode(row["human_text"], add_special_tokens=False)
                    continuation = continuation[:1+index%max(1,len(continuation))]
                ids = torch.tensor([prefix+continuation], device="cuda")
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                logits = model(ids, use_cache=False).logits[:, -1]
                end.record();torch.cuda.synchronize()
                model_ms = begin.elapsed_time(end)
                probabilities = (logits.float()/settings["temperature"]).softmax(-1)[0]
                values, tokens = probabilities.sort(descending=True)
                active = values.cumsum(-1)-values <= settings["top_p"]
                values, tokens = values[active].double(), tokens[active]
                values /= values.sum()
                context = ids[:, -3:]
                key = config["watermark"]["secret_key"]
                uniforms = [keyed_uniform(context, tokens[None], k, "hires") for k in [key,key+DUAL_KEY_OFFSET]]
                gaussian = sum(torch.special.ndtri(u.clamp(1.e-15,1-1.e-15)) for u in uniforms)[0]/math.sqrt(2)
                native_gaussian = sum(torch.special.ndtri(u.clamp(1.e-7,1-1.e-7)) for u in uniforms)[0]/math.sqrt(2)
                rho = 0. if repeated else config["watermark"]["rho"]
                sigma = math.sqrt(1-rho*rho)
                counts = {name: torch.zeros_like(values) for name in ["ideal","native"]}
                generator = torch.Generator(device="cuda").manual_seed(config["seed"]+9000000+index)
                normal = torch.distributions.Normal(torch.zeros((),device="cuda",dtype=torch.float64),torch.ones((),device="cuda",dtype=torch.float64))
                disagreements = 0
                for start in range(0,spec["reference_draws"],spec["reference_chunk"]):
                    size = min(spec["reference_chunk"],spec["reference_draws"]-start)
                    noise = torch.randn((size,len(values)),device="cuda",dtype=torch.float64,generator=generator)
                    ideal = (-torch.special.log_ndtr(rho*gaussian+sigma*noise)/values).argmin(-1)
                    native = (-normal.cdf(rho*native_gaussian+sigma*noise).clamp(1.e-7,1-1.e-7).log()/values).argmin(-1)
                    for name,winners in [("ideal",ideal),("native",native)]:
                        counts[name] += torch.bincount(winners,minlength=len(values))
                    disagreements += int((ideal!=native).sum())
                references = {name:count/spec["reference_draws"] for name,count in counts.items()}
                estimates = []
                for draws in spec["draw_counts"]:
                    for repeat in range(spec["repeats"]):
                        generator = torch.Generator(device="cuda").manual_seed(config["seed"]+10000000+index*100000+draws*10+repeat)
                        begin.record()
                        q, receipt = conditional_race_probabilities(values,gaussian,rho,draws,generator)
                        end.record();torch.cuda.synchronize()
                        best = q.topk(min(settings["k"],len(q))).indices
                        estimates.append({"draws": draws,"repeat":repeat,"cuda_ms":begin.elapsed_time(end),
                            "raw_sum":receipt["raw_sum"],"omitted_top10_estimated_mass":float(1-q[best].sum()),
                            "omitted_top10_native_reference_mass":float(1-references["native"][best].sum()),
                            **{name+"_tv":float(.5*(q-frequencies).abs().sum()) for name,frequencies in references.items()},
                            **{name+"_max_abs":float((q-frequencies).abs().max()) for name,frequencies in references.items()}})
                rows.append({"calibration_index":index,"id":row["id"],"active_support":len(values),
                    "prefix_case":case,"effective_rho":rho,
                    "model_forward_cuda_ms":model_ms,"reference_draws":spec["reference_draws"],
                    "native_ideal_disagreement_count":disagreements,"estimates":estimates})
                save(output,{"signature":config["signature"],"rows":rows,"scope":spec["scope"]})
                print("probability audit",len(rows),"of",len(spec["prefix_indices"]),flush=True)
                pause(stop)
            save(root/"audit_complete.json",{"signature":config["signature"],"results_sha256":sha(output)})
            return
        bounds_path = root / "confidence.json"
        if not bounds_path.exists():
            interfaces = {"pre":None,"conditional":None,**config.get("exposure_guard",{}).get("coverage_loss",{})}
            values = {name:[] for name in interfaces}
            for index, row in enumerate(config["data"]["calibration"]):
                text = prompt(tokenizer, config["rewrite_instruction"], row["human_text"])
                prefix = tokenizer.encode(text, add_special_tokens=False)
                continuation = tokenizer.encode(row["human_text"], add_special_tokens=False)
                length = 1 + index % max(1, len(continuation))
                ids = torch.tensor([prefix + continuation[:length]], device="cuda")
                logits = model(ids, use_cache=False).logits[:, -1]
                for name,guard_loss in interfaces.items():
                    response, _ = public_response(logits, ids[:, -3:], name!="pre", config["seed"] + index,guard_loss)
                    values[name].append(float(confidence(response)[0]))
            bounds = {name: {"lower": min(v), "upper": max(v), "values": v} for name, v in values.items()}
            if any(r["lower"] >= r["upper"] for r in bounds.values()):
                raise ValueError("Degenerate confidence calibration")
            save(bounds_path, bounds)
        bounds = json.loads(bounds_path.read_text())
        original_path = root / "originals.json"
        originals = json.loads(original_path.read_text()) if original_path.exists() else []
        sampler = build_generator(model, tokenizer, textseal2_config(config["watermark"]["secret_key"]))
        for index in range(len(originals), len(config["data"]["evaluation"])):
            row = config["data"]["evaluation"][index]
            text = prompt(tokenizer, config["original_instruction"], row["question"])
            seed_everything(config["seed"] + 10000 + index)
            tokens = sampler.generate([text], max_gen_len=settings["max_tokens"], temperature=settings["temperature"],
                                      top_p=settings["top_p"], return_ids=True)[0]
            originals.append({"id": row["id"], "question": row["question"], "prompt": text,
                              "tokens": tokens, "text": tokenizer.decode(tokens, skip_special_tokens=True),
                              "length_capped": len(tokens) == settings["max_tokens"]})
            save(original_path, originals)
            print("original", len(originals), flush=True)
            pause(stop)
        arms = config.get("attack_arms", ARMS)
        for arm in arms:
            filename = root / (arm + ".json")
            saved = json.loads(filename.read_text()) if filename.exists() else []
            expected = [(r["id"], attempt) for r in originals for attempt in range(settings["attempts"])]
            if [(r["id"], r["attempt"]) for r in saved] != expected[:len(saved)]:
                raise ValueError("Attack attempt restart alignment changed")
            for number in range(len(saved), len(expected)):
                index, attempt = divmod(number, settings["attempts"])
                row = originals[index]
                text = prompt(tokenizer, config["rewrite_instruction"], row["text"])
                prefix = tokenizer.encode(text, add_special_tokens=False)
                ids = torch.tensor([prefix], device="cuda")
                cache, reference_cache = configured_cache(model), configured_cache(reference)
                generated, weights, integrals, seen, guards = [], [], [], set(), []
                evidence_floors,guard_membership=[],[]
                order = settings.get("source_ngram_order")
                source_blocks = source_ngram_completions(tokenizer.encode(row["text"], add_special_tokens=False), order) if order else {}
                fallback_count = 0
                conditional = arm.startswith(("conditional","guard"))
                interface = arm.split("_",1)[0] if arm.startswith("guard") else "conditional" if conditional else "pre"
                guard_loss = config.get("exposure_guard",{}).get("coverage_loss",{}).get(interface)
                current_bounds = bounds[interface]
                seed_everything(config["seed"] + 20000 + number)
                eos = model.generation_config.eos_token_id
                eos = set((eos if isinstance(eos, list) else [eos]) + [tokenizer.eos_token_id])
                for position in range(settings["max_tokens"]):
                    context = torch.tensor([(prefix + generated)[-3:]], device="cuda")
                    window = tuple(context[0].tolist())
                    response, integral = None,None
                    if arm!="reference_only":
                        result = model(ids, past_key_values=cache, use_cache=True)
                        cache = result.past_key_values
                        response, integral = public_response(result.logits[:, -1], context,
                            conditional and window not in seen, config["seed"] + 1000000 + number * settings["max_tokens"] + position,guard_loss)
                    seen.add(window)
                    if integral:
                        integrals.append(integral["raw_sum"])
                        if integral.get("guard"):
                            guards.append(integral["guard"])
                    # The blind client below sees only top-k responses, never the key.
                    reference_response = None
                    reference_result = None
                    if arm.endswith(("smooth", "ratio")) or arm=="reference_only":
                        reference_result = reference(ids, past_key_values=reference_cache, use_cache=True)
                        reference_cache = reference_result.past_key_values
                        if arm.endswith("smooth") or arm=="reference_only":
                            reference_response, _ = public_response(reference_result.logits[:, -1], context)
                    forbidden = source_blocks.get(tuple(generated[-(order-1):]), set()) if order and len(generated) >= order-1 else set()
                    if arm=="reference_only":
                        choice,_,fallback = constrained_api_choice(reference_response,forbidden)
                        fallback_count += int(fallback[0])
                    elif arm.endswith("ratio"):
                        reference_probabilities = (reference_result.logits[:, -1].float()/settings["temperature"]).softmax(-1)
                        choice, fallback = ratio_api_choice(response, reference_probabilities.gather(-1, response.ids),
                            forbidden, settings["ratio_exponents"][attempt])
                        fallback_count += int(fallback[0])
                    elif order:
                        choice, weight, fallback = constrained_api_choice(response, forbidden, reference_response,
                            current_bounds["lower"], current_bounds["upper"], settings["alpha"])
                        fallback_count += int(fallback[0])
                        if weight is not None:
                            weights.append(float(weight[0]))
                    elif reference_response is not None:
                        choice, weight = smooth(response, reference_response, current_bounds["lower"], current_bounds["upper"])
                        weights.append(float(weight[0]))
                    else:
                        choice = resample(response)
                    token = int(choice[0])
                    if token in eos:
                        break
                    generated.append(token)
                    guard=(integral or {}).get("guard")
                    evidence_floors.append(guard["score_floor"] if guard else None)
                    guard_membership.append(bool((response.ids==token).any()) if guard else False)
                    ids = choice[:, None]
                saved.append({"id": row["id"], "attempt": attempt, "tokens": generated,
                              "text": tokenizer.decode(generated, skip_special_tokens=True),
                              "length_capped": len(generated) == settings["max_tokens"],
                              "source_blocking_fallbacks": fallback_count,
                              "ratio_exponent": settings["ratio_exponents"][attempt] if arm.endswith("ratio") else None,
                              "guarded_queries":len(guards),
                              "evidence_floors":evidence_floors,"guard_support_membership":guard_membership,
                              "mean_guard_score_floor":sum(g["score_floor"]for g in guards)/len(guards) if guards else None,
                              "mean_guard_relative_mass_loss":sum(g["relative_mass_loss"]for g in guards)/len(guards) if guards else None,
                              "mean_guard_retained_mass":sum(g["retained_mass"]for g in guards)/len(guards) if guards else None,
                              "mean_target_weight": sum(weights)/len(weights) if weights else None,
                              "mean_integral_sum": sum(integrals)/len(integrals) if integrals else None})
                save(filename, saved)
                print(arm, len(saved), flush=True)
                pause(stop)
        save(root / "generation_complete.json", {"signature": config["signature"],
            "outputs": {str(p.name): sha(p) for p in [original_path] + [root/(arm+".json") for arm in arms]},
            "status": "All attempts frozen; quality selection and owner scoring still required."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "audit"])
    parser.add_argument("manifest", nargs="?")
    parser.add_argument("--reuse-parent")
    parser.add_argument("--block-source-ngrams", action="store_true")
    parser.add_argument("--ratio-attack", action="store_true")
    parser.add_argument("--confirmation-shard", type=int, choices=[0, 1])
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--audit-attack-prefixes", action="store_true")
    parser.add_argument("--support-guard", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.reuse_parent, args.block_source_ngrams, args.ratio_attack, args.confirmation_shard, args.audit, args.audit_attack_prefixes,args.support_guard)
    else:
        from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
        try:
            run(args.manifest, args.action == "audit")
        except CheckpointPause as error:
            print(str(error), flush=True)
            sys.exit(75)
