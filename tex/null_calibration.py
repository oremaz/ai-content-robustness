"""Audit the unchanged native TextSeal2 threshold on separate human prefixes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

from development_summary import ROOT, read, save, sha

sys.path.insert(0, str(ROOT/"scripts"))
from publish_watermark_results import binomial_ci


def identity(value):
    return hashlib.sha256(value.encode()).hexdigest()


def run(parent, confirmations):
    config = read(parent)
    prior = read(config["parent"])
    retained = read(prior["data"])["text"]
    excluded = {r["id"] for rows in retained.values() for r in rows}
    used_human = set()
    for current in [config]+[read(path) for path in confirmations]:
        for rows in current["data"].values():
            excluded.update(r["id"] for r in rows)
            used_human.update(identity(r["human_text"]) for r in rows)
    pool, seen_ids, seen_texts = [], set(excluded), set(used_human)
    data = ROOT/"data/hc3-all.jsonl"
    for line in data.read_text().splitlines():
        row = json.loads(line)
        question = row["question"].strip()
        key = identity(question.lower())
        if key in seen_ids or not 30 <= len(question) <= 1500 or not row["human_answers"]:
            continue
        seen_ids.add(key)
        if not 65 <= int(key[:12],16)%100 < 90:
            continue
        text = " ".join(row["human_answers"][0].split()[:100])
        digest = identity(text)
        if not text or digest in seen_texts:
            continue
        seen_texts.add(digest)
        pool.append({"id":key,"human_text":text,"text_sha256":digest})
    pool.sort(key=lambda r:identity("2026110803:"+r["id"]))
    if len(pool)<4096:
        raise ValueError("Insufficient independent human prefixes for the frozen 4096-text audit")
    rows=pool[:4096]
    root=ROOT/"results/research_workflow/paper_writing"/(
        "text-null-audit-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    root.mkdir()
    shutil.copyfile(__file__,root/"analysis_source.py")
    manifest={"parent":str(Path(parent).resolve()),"parent_sha256":sha(parent),
              "confirmation_manifests":{str(Path(p).resolve()):sha(p) for p in confirmations},
              "data_sha256":sha(data),"rows":rows,"source_sha256":sha(root/"analysis_source.py"),
              "n":4096,"nominal_p_value_threshold":.001,"compute":"CPU only; no new GPU allocation",
              "protocol":"Frozen before scoring. Hash buckets65..89, excluded all retained/development/confirmation/calibration question IDs and known human-prefix hashes; deduplicated questions and first100-word human prefixes. One existing key, unchanged native v2 detector and threshold. No calibration or threshold fitting."}
    save(root/"manifest.json",manifest)
    print(root,flush=True)
    sys.path.insert(0,config["textseal2_source"])
    with patch("importlib.metadata.packages_distributions",return_value=config["package_map"]):
        import torch
        from transformers import AutoTokenizer
        from textseal.watermarking.detector import build_detector
        from textseal.watermarking.core import copula_scores
    from ai_content_detector.watermarking.instruct_text import textseal2_config
    tokenizer=AutoTokenizer.from_pretrained(config["generator"]["path"],local_files_only=True)
    detector=build_detector(tokenizer,textseal2_config(config["watermark"]["secret_key"]))
    ids=tokenizer([r["human_text"] for r in rows],add_special_tokens=False)["input_ids"]
    width=detector.ngram

    def vector_scores(documents):
        pairs=[list(dict.fromkeys(tuple(tokens[t-width:t+1]) for t in range(width,len(tokens))))
               for tokens in documents]
        flat=[pair for group in pairs for pair in group]
        values=[]
        for start in range(0,len(flat),65536):
            batch=torch.tensor(flat[start:start+65536],dtype=torch.long)
            values.extend(copula_scores(batch[:,:-1],batch[:,-1],
                [detector.wm_args.key_a,detector.wm_args.key_b],detector.wm_args.prf).tolist())
        result=[]
        start=0
        for group in pairs:
            result.append(values[start:start+len(group)]);start+=len(group)
        return result

    probes=[[],[1],[1,2,3],[1,2,3,4]*4]+ids[::512]
    for tokens,batched in zip(probes,vector_scores(probes)):
        native=detector.get_scores_by_t([tokens],scoring_method="v2",seen_windows=set())[0]
        if len(native)!=len(batched) or any(not math.isclose(a,b,rel_tol=1.e-12,abs_tol=1.e-12)
                                          for a,b in zip(native,batched)):
            raise ValueError("Batched native score differs from the released scalar detector")
    print("Native scalar/vector checks passed",flush=True)
    scores=vector_scores(ids)
    per_text=[{"id":row["id"],"tokens":len(tokens),"scored_pairs":len(values),
               "p_value":detector.get_pvalue(math.fsum(values),len(values))}
              for row,tokens,values in zip(rows,ids,scores)]
    save(root/"scores.json",per_text)
    count=sum(r["p_value"]<=.001 for r in per_text)
    receipt={"manifest":str(root/"manifest.json"),"manifest_sha256":sha(root/"manifest.json"),
             "scores_sha256":sha(root/"scores.json"),"n":len(rows),"detected":count,
             "rate":count/len(rows),"exact_binomial_ci95":binomial_ci(count,len(rows)),
             "no_scored_pairs":sum(not r["scored_pairs"] for r in per_text),
             "scalar_vector_probes":len(probes),"nominal_p_value_threshold":.001,
             "scope":manifest["protocol"],"compute":manifest["compute"]}
    save(root/"complete.json",receipt)
    save(ROOT/"tex/provenance/textseal2_human_null_audit.json",receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent")
    parser.add_argument("confirmations",nargs=2)
    args=parser.parse_args()
    run(args.parent,args.confirmations)
