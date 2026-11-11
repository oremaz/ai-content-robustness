"""TextSeal2 and DistSeal-inspired instruction-model experiments."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile

from bira_official import save, sha, local_models
import bira_replication as bira

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT/"results/research_workflow/watermarking/resources"
GENERATOR = "Qwen/Qwen3.5-4B"
REWRITER = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CHILD = None


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_runtime():
    archive = RESOURCES/"bira_runtime_py310.tar.gz"
    destination = RESOURCES/"bira_runtime_py310"
    with (RESOURCES/"bira_runtime_py310.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        signature = sha(archive)
        marker = destination/"runtime_complete.json"
        if marker.exists() and read(marker).get("archive_sha256") == signature:
            if (destination/"bin/python").is_file():
                return destination/"bin/python", signature
        if destination.exists():
            raise ValueError("Existing shared BIRA runtime is incomplete or differs from its archive")
        with tarfile.open(archive, "r:gz") as stream:
            members = stream.getmembers()
        storage = os.statvfs(RESOURCES)
        if storage.f_bavail*storage.f_frsize < sum(m.size for m in members)+2*2**30:
            raise ValueError("Insufficient project disk space for the shared BIRA runtime")
        if storage.f_favail < len(members)+2000:
            raise ValueError("Insufficient project inodes for the shared BIRA runtime")
        temporary = Path(tempfile.mkdtemp(prefix="bira_runtime_build_", dir=RESOURCES))
        try:
            subprocess.run(["tar", "--no-same-owner", "-xzf", str(archive), "-C", str(temporary)], check=True)
            if not (temporary/"bin/python").is_file():
                raise ValueError("The extracted BIRA interpreter is missing")
            save(temporary/"runtime_complete.json", {"archive_sha256": signature, "entries": len(members)})
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return destination/"bin/python", signature


def prepare():
    paper_python, runtime_signature = prepare_runtime()
    from transformers import AutoTokenizer
    from ai_content_detector.watermarking.instruct_text import prepare_questions, sampling_settings
    assets = read(RESOURCES/"textseal2_models.json")
    assert assets["generator"]["name"] == GENERATOR
    assert assets["reference"]["name"] == "Qwen/Qwen3.5-0.8B"
    models = {k: v for k, v in read(RESOURCES/"bira_models.json").items()
              if k in (REWRITER, "cross-encoder/nli-deberta-v3-large", "sentence-transformers/all-mpnet-base-v2", "mistralai/Mistral-7B-v0.1")}
    if REWRITER not in models:
        raise ValueError("Missing independent Llama BIRA checkpoint")
    models.update({item["name"]: item for item in assets.values()})
    for item in models.values():
        p = Path(item["path"])
        if not (p/"config.json").exists() or not (list(p.glob("*.safetensors"))+list(p.glob("pytorch_model*.bin"))):
            raise ValueError(f"Incomplete checkpoint: {p}")
        for index in p.glob("*.safetensors.index.json"):
            if not all((p/name).is_file() for name in set(read(index)["weight_map"].values())):
                raise ValueError(f"Missing model shards: {p}")
    revision = subprocess.check_output(["git", "-C", str(RESOURCES/"textseal2"), "rev-parse", "HEAD"], text=True).strip()
    if revision != "0b76bc64013081c570b8633d3d3585e3ee9a495a":
        raise ValueError("TextSeal2 source revision differs from the reviewed release")
    directory = ROOT/"results/research_workflow/text_instruct"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    source = directory/"source"
    shutil.copytree(ROOT/"ai_content_detector", source/"ai_content_detector", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (source/"scripts").mkdir()
    for name in ("instruct_watermark_experiments.py", "run_instruct_watermarks.slurm", "bira_official.py", "bira_replication.py", "textseal_api_attacks.py"):
        shutil.copyfile(ROOT/"scripts"/name, source/"scripts"/name)
    for name, origin in (("released_textseal2", RESOURCES/"textseal2"), ("upstream", RESOURCES/"bira")):
        shutil.copytree(origin, directory/name, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    tokenizer = AutoTokenizer.from_pretrained(assets["generator"]["path"], local_files_only=True)
    data = prepare_questions(ROOT/"data/hc3-all.jsonl", tokenizer, 20261103)
    save(directory/"partitions.json", data)
    for kind in ("textseal2", "latent"):
        (directory/kind).mkdir()
    cfg = bira.read_yaml(RESOURCES/"bira/model_config/llama3.1-8b.yaml")
    cfg.update(name=REWRITER, use_fp16=True)
    save(directory/"llama.yaml", cfg)
    latent_dir = directory/"latent_training"
    latent_dir.mkdir()
    import numpy as np
    latent = {"directory": str(latent_dir), "source": str(source), "method": "distseal_latent_text",
        "qwen": assets["generator"], "data": str(directory/"partitions.json"), "seed": 20261103,
        "distillation_stage": "latent_distill", "distilled_model": "latent_distill/model",
        "latent_training": {"steps": 7500, "validate_every": 500, "batch": 2, "length": 128,
            "bits": 32, "width": 256, "strength": .65, "temperature": 1., "lr": 1e-4,
            "kl_budget": .15, "minimum_bit_accuracy": .5,
            "message": np.random.default_rng(20261103).integers(2, size=32).tolist()},
        "text_training": {"steps": 8000, "clean_steps": 1500, "accumulation": 8, "length": 256,
            "lr": 3e-5, "clean_lr": 1e-5, "lora_rank": 128, "mode": "lora"}, "jobs": {}}
    latent["protocol_sha256"] = digest(latent)
    save(directory/"latent_manifest.json", latent)
    config = {"directory": str(directory), "source": str(source), "upstream": str(directory/"upstream"),
        "textseal2_source": str(directory/"released_textseal2"), "textseal2_revision": revision,
        "paper": str(RESOURCES/"textseal2_paper/textseal2.pdf"), "paper_sha256": sha(RESOURCES/"textseal2_paper/textseal2.pdf"),
        "models": models, "generator": assets["generator"], "reference": assets["reference"],
        "main_python": str(ROOT/".venv/bin/python"), "runtime_archive": str(RESOURCES/"bira_runtime_py310.tar.gz"),
        "paper_python": str(paper_python), "runtime_archive_sha256": runtime_signature,
        "data": str(directory/"partitions.json"), "latent_manifest": str(directory/"latent_manifest.json"),
        "sampling": sampling_settings(), "seed": 20261103,
        "cases": [{"name": "textseal2", "kind": "textseal2", "rewriter": REWRITER, "runtime": "paper"},
                  {"name": "latent", "kind": "latent", "rewriter": REWRITER, "runtime": "paper"}],
        "watermark": {"watermark_type": "textseal2", "secret_key": 42, "rho": .9, "ngram": 3, "prf": "hires", "repeated_windows": "skip"},
        "scope": "TextSeal2 target-only decoding and DistSeal-inspired latent text teacher/student on instruct Qwen. No TextSeal1 or official DistSeal text-reproduction claim. BIRA uses only unrelated-family Llama weights.",
        "quality_gate_enforced": False,
        "input_hashes": {str(p): sha(p) for p in (directory/"partitions.json", directory/"latent_manifest.json", directory/"llama.yaml", paper_python.parent.parent/"runtime_complete.json")},
        "frozen_hashes": {str(p.relative_to(directory)): sha(p) for base in (source, directory/"released_textseal2", directory/"upstream") for p in base.rglob("*") if p.is_file()}}
    config["signature"] = digest(config)
    save(directory/"manifest.json", config)
    print(directory/"manifest.json", flush=True)


def setup(config):
    sys.path.insert(0, config["source"])
    sys.path.insert(0, str(Path(config["source"])/"scripts"))
    sys.path.insert(0, config["textseal2_source"])
    if "textseal" in sys.modules:
        path = Path(sys.modules["textseal"].__file__).resolve()
        if not path.is_relative_to(Path(config["textseal2_source"])):
            raise ValueError("An older TextSeal version was imported")
    os.chdir(config["upstream"])


def sample_records(config, model, tokenizer, rows, path, stop, marked=False):
    import torch
    from dataclasses import replace
    from textseal.watermarking.generator import build_generator
    from ai_content_detector.watermarking.instruct_text import textseal2_config
    data = bira.rows_file(path, rows, config["signature"])
    wm = textseal2_config() if marked else replace(textseal2_config(), watermark_type="none")
    sampler = build_generator(model, tokenizer, wm)
    settings = config["sampling"]
    for start in range(len(data), len(rows), settings["batch"]):
        batch = rows[start:start+settings["batch"]]
        torch.manual_seed(config["seed"]+start)
        ids = sampler.generate([x["prompt"] for x in batch], max_gen_len=settings["max_tokens"],
            temperature=settings["temperature"], top_p=settings["top_p"], return_ids=True)
        for row, tokens in zip(batch, ids):
            data.append({"id": row["id"], "question": row["question"], "prompt": row["prompt"],
                "text": tokenizer.decode(tokens, skip_special_tokens=False), "tokens": tokens,
                "generated_tokens": len(tokens), "length_capped": len(tokens)==settings["max_tokens"]})
        bira.save_rows(path, data, config["signature"])
        print(path.parent.name, path.name, len(data), flush=True)
        stop.check()
    return data


def generate(config, kind, stop):
    import torch
    from ai_content_detector.watermarking.text import load_model
    from ai_content_detector.watermarking.latent_text import load_pair, embedded
    root = Path(config["directory"]);out=root/kind;data=read(config["data"])["text"]
    model, tokenizer = load_model(config["generator"]["path"])
    sample_records(config, model, tokenizer, data["validation"], out/"clean_validation.json", stop)
    sample_records(config, model, tokenizer, data["test"], out/"clean.json", stop)
    if kind == "textseal2":
        samples = sample_records(config, model, tokenizer, data["test"], out/"generated.json", stop, marked=True)
    else:
        latent = read(config["latent_manifest"])
        embedder, extractor, _ = load_pair(latent)
        del extractor
        with embedded(model, embedder, latent["latent_training"]["message"]):
            sample_records(config, model, tokenizer, data["test"], out/"teacher.json", stop)
        del model, embedder
        gc.collect();torch.cuda.empty_cache()
        model, tokenizer = load_model(Path(latent["directory"])/latent["distilled_model"])
        samples = sample_records(config, model, tokenizer, data["test"], out/"generated.json", stop)
    save(out/"inputs.json", samples)
    humans=[{"id":r["id"],"text":r["human_text"]} for r in data["test"]]
    save(out/"humans.json", humans)


def probabilities(config, kind, stop):
    import torch
    import textseal_api_attacks as attacks
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
    out=Path(config["directory"])/kind;latent=read(config["latent_manifest"])
    source=config["generator"]["path"] if kind=="textseal2" else str(Path(latent["directory"])/latent["distilled_model"])
    save(out/"watermarked.json", read(out/"inputs.json"))
    settings=config["sampling"]
    recipe={"directory":str(out),"original":str(out),"owner":{"qwen":{"path":source}},
        "reference":config["reference"]["path"],"data":config["data"],"seed":config["seed"],
        "batch":settings["batch"],"k":10,"alpha":1.,"confidence_prefixes":200,
        "generation_tokens":settings["max_tokens"],"temperature":settings["temperature"],"top_p":settings["top_p"],
        "protocol_sha256":config["signature"]}
    try:
        with torch.inference_mode():attacks.probability_run(recipe, stop)
    except CheckpointPause as error:
        raise bira.Pause(str(error)) from error


def owner(config, kind, stop):
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from transformers import AutoTokenizer
    root=Path(config["directory"]);out=root/kind;case=next(x for x in config["cases"] if x["name"]==kind)
    arms=bira.freeze_attacks(config,case)
    tokenizer=AutoTokenizer.from_pretrained(config["generator"]["path"],local_files_only=True)
    data=read(config["data"])["text"]
    if kind=="textseal2":
        from textseal.watermarking.detector import build_detector
        from ai_content_detector.watermarking.instruct_text import textseal2_config,textseal2_score
        detector=build_detector(tokenizer,textseal2_config())
        score_one=lambda text:textseal2_score(detector,text)
        thresholds={"fpr001":3.,"fpr01":2.}
    else:
        from ai_content_detector.watermarking.latent_text import load_pair,score
        latent=read(config["latent_manifest"]);embedder,extractor,_=load_pair(latent);del embedder
        def score_one(text):
            scores,records=score(extractor,tokenizer,[text],latent["latent_training"]["message"])
            return records[0]
        thresholds=None
    pools={"original":read(out/"inputs.json"),"clean":read(out/"clean.json")["rows"],"human":read(out/"humans.json")}
    if kind=="latent":
        pools.update(teacher=read(out/"teacher.json")["rows"],calibration=read(out/"clean_validation.json")["rows"],
                     human_calibration=[{"id":x["id"],"text":x["human_text"]} for x in data["validation"]])
    pools.update({arm:read(out/(arm+".json"))["rows"] for arm in arms})
    scores={}
    for name,rows in pools.items():
        path=out/(name+"_scores.json");saved=bira.rows_file(path,rows,config["signature"])
        for row in rows[len(saved):]:
            with torch.inference_mode():result=score_one(row["text"])
            saved.append(dict(result,id=row["id"]))
            bira.save_rows(path,saved,config["signature"]);stop.check()
        scores[name]=saved
    if thresholds is None:
        thresholds={name:max(float(np.quantile([x["score"] for x in scores[group]],q,method="higher"))
                            for group in ("calibration","human_calibration")) for name,q in (("fpr001",.999),("fpr01",.99))}
    detected=lambda value,threshold: value>=threshold if kind=="textseal2" else value>threshold
    results={}
    for name in ["original","clean"]+(["teacher"] if kind=="latent" else [])+arms:
        values=[x["score"] for x in scores[name]];null=[x["score"] for x in scores["human"]]
        results[name]={"n":len(values),"eligible":sum(x["eligible"] for x in scores[name]),
            "auroc":float(roc_auc_score([0]*len(null)+[1]*len(values),null+values)),
            "operating_points":{label:{"detected":sum(detected(x,threshold) for x in values),"detection_rate":sum(detected(x,threshold) for x in values)/len(values),
                "human_false_positives":sum(detected(x,threshold) for x in null),"human_n":len(null)} for label,threshold in thresholds.items()}}
    save(out/"detection.json",{"thresholds":thresholds,"conditions":results,
        "policy":"Fixed unweighted Gamma p-value thresholds" if kind=="textseal2" else "Frozen thresholds from disjoint clean and human validation; 256 null documents cannot precisely calibrate 0.1% FPR.",
        "limitations":"Raw detector evasion is separate from quality, eligible length and preservation. No threshold fitting on attacks or test data."})
    for name,expected in read(out/"attacks_complete.json").items():
        if sha(out/(name+".json"))!=expected:raise ValueError("Attack artifact changed during scoring")


def preflight(config,stop):
    import torch
    from ai_content_detector.watermarking.text import load_model
    from ai_content_detector.watermarking.latent_text import architecture,make_pair,train_batch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<70*2**30:
        raise ValueError("An 80GB-class allocated GPU is required")
    model,tokenizer=load_model(config["generator"]["path"])
    other=AutoTokenizer.from_pretrained(config["reference"]["path"],local_files_only=True)
    if tokenizer.get_vocab()!=other.get_vocab():raise ValueError("Smoothing reference vocabulary differs")
    rows=read(config["data"])["text"]["validation"][:20];root=Path(config["directory"])
    sample_records(config,model,tokenizer,rows,root/"sanity_clean.json",stop)
    sample_records(config,model,tokenizer,rows,root/"sanity_textseal2.json",stop,marked=True)
    from textseal.watermarking.detector import build_detector
    from ai_content_detector.watermarking.instruct_text import textseal2_config,textseal2_score
    detector=build_detector(tokenizer,textseal2_config())
    detection={}
    for name in ("sanity_clean","sanity_textseal2"):
        values=[textseal2_score(detector,x["text"]) for x in read(root/(name+".json"))["rows"]]
        detection[name]={"n":len(values),"detected_p001":sum(x["p_value"]<=.001 for x in values),
                         "scored_pairs":[x["scored_pairs"] for x in values]}
    if not any(detection["sanity_textseal2"]["scored_pairs"]):raise ValueError("TextSeal2 generated no scorable output")
    latent=read(config["latent_manifest"]);settings={**latent["latent_training"],"length":8,"batch":1}
    embedder,extractor=make_pair(architecture(model,settings))
    rng=torch.Generator(device="cuda").manual_seed(config["seed"])
    bce,kl,accuracy=train_batch(model,tokenizer,embedder,extractor,rows[:1],settings,rng)
    (bce+kl).backward()
    norm=torch.nn.utils.clip_grad_norm_(list(embedder.parameters())+list(extractor.parameters()),1.)
    if not torch.isfinite(norm) or embedder.up.weight.grad.abs().sum()==0:
        raise ValueError("Instruction-response watermark gradient failed")
    save(root/"preflight.json",{"cuda":torch.cuda.get_device_name(),"model":GENERATOR,"bira_model":REWRITER,
        "bce":float(bce),"kl":float(kl),"bit_accuracy":float(accuracy),"gradient_norm":float(norm),
        "quality_is_not_a_gate":True,"samples_per_sanity_condition":20,"detection":detection})


def phase(config,name):
    from ai_content_detector.rl_evasion.checkpointing import CheckpointPause
    from ai_content_detector.rl_evasion.config import seed_everything
    stop=bira.Stop()
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,stop.request)
    root=Path(config["directory"]);receipt=root/(name+"_receipt.json")
    with (root/(name+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);bira.verify(config)
        if receipt.exists() and read(receipt).get("status")=="completed":return
        setup(config);seed_everything(config["seed"])
        save(receipt,{"status":"running","job_id":os.environ.get("SLURM_JOB_ID")})
        try:
            with local_models(config["models"]):
                if name=="preflight":preflight(config,stop)
                elif name in ("teacher_train","student_train"):
                    from ai_content_detector.watermarking import latent_text
                    manifest=read(config["latent_manifest"])
                    if name=="teacher_train":latent_text.train(manifest,stop)
                    else:latent_text.fit(manifest,stop)
                else:
                    kind,stage=name.split(":",1);case=next(x for x in config["cases"] if x["name"]==kind)
                    if stage=="generate":generate(config,kind,stop)
                    elif stage=="probabilities":probabilities(config,kind,stop)
                    elif stage in ("vanilla","bira"):bira.attack_rows(config,case,stage,stop)
                    elif stage=="owner":owner(config,kind,stop)
                    elif stage=="quality":bira.quality(config,case,stop)
                    else:raise ValueError(name)
            stop.check();save(receipt,{"status":"completed","job_id":os.environ.get("SLURM_JOB_ID")})
        except (bira.Pause,CheckpointPause):
            save(receipt,{"status":"paused","signal":stop.signum});raise SystemExit(75 if stop.signum==signal.SIGUSR1 else 0)
        except BaseException as error:
            save(receipt,{"status":"failed","error":repr(error)});raise


def run(config,path,task):
    global CHILD
    stop=bira.Stop()
    def forward(signum,frame):
        stop.request(signum,frame)
        if CHILD is not None:
            try:CHILD.send_signal(signum)
            except ProcessLookupError:pass
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,forward)
    names=["preflight"] if task=="preflight" else (["teacher_train","student_train"] if task=="latent" else [])
    if task!="preflight":names += [task+":"+s for s in ("generate","probabilities","vanilla","bira","owner","quality")]
    for name in names:
        runtime="paper" if name.endswith((":vanilla",":bira",":quality")) else "main"
        python=os.environ["BIRA_PAPER_PYTHON"] if runtime=="paper" else config["main_python"]
        CHILD=subprocess.Popen([python,str(Path(config["source"])/"scripts/instruct_watermark_experiments.py"),"phase",str(path),name])
        code=CHILD.wait();CHILD=None
        if stop.requested:raise SystemExit(75 if stop.signum==signal.SIGUSR1 else 0)
        if code:raise SystemExit(code)


def submit(path):
    path=Path(path).resolve();config=read(path);bira.verify(config);jobs=config.get("jobs",{})
    for name in ("preflight","textseal2","latent"):
        if name in jobs:continue
        args=["sbatch","--parsable","--job-name=instruct-"+name,"--output="+str(Path(config["directory"])/(name+"-%j.out"))]
        if name!="preflight":args += ["--dependency=afterok:"+jobs["preflight"]]
        args += [str(Path(config["source"])/"scripts/run_instruct_watermarks.slurm"),str(path),name]
        jobs[name]=subprocess.check_output(args,text=True).strip().split(";")[0];config["jobs"]=jobs;save(path,config)
    print(json.dumps(jobs,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","submit","run","phase"));parser.add_argument("manifest",nargs="?");parser.add_argument("task",nargs="?")
    args=parser.parse_args()
    if args.action=="prepare":prepare()
    elif args.action=="submit":submit(args.manifest)
    elif args.action=="phase":phase(read(args.manifest),args.task)
    else:run(read(args.manifest),args.manifest,args.task)
