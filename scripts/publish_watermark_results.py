"""Validate the portable publication or export historical Markdown separately."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import re
import statistics

from PIL import Image, __version__ as pillow_version

from report_research import table
from watermark_report_assets import (
    font, full_text, image_grid, sample_positions, sha, text_grid, validate_publication,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"results/research_workflow"
TEXT = RUNS/"text_instruct/20261103T231534057688Z"
JINA = RUNS/"jina_similarity/20261104T081330240270Z"
IMAGES = RUNS/"watermarking/20261026T001727478096Z"
WEIGHTS = RUNS/"watermark_removal/20261026T110140724565Z"
NFPA = RUNS/"closed_image/20261104T082631753968Z"
QUALITY = RUNS/"image_quality/20261104T153348508006Z"
PAPER = RUNS/"paper_bira_retained"
METHODS = {"vanilla":"Ordinary paraphrasing", "bira":"BIRA", "direct_topk":"Direct top-10 resampling", "smoothing":"Confidence smoothing (adapted)"}
IMAGE_LABELS = {
    "weight_unmodified_supplied":"Unmodified watermarked model", "weight_merge_0.75":"Merge: 75% watermarked",
    "weight_merge_0.5":"Merge: 50% watermarked", "weight_merge_0.25":"Merge: 25% watermarked",
    "weight_merge_0":"Original clean model replacement", "weight_clean_lora_500":"Clean LoRA: 500 steps",
    "weight_clean_lora_2500":"Clean LoRA: 2,500 steps", "weight_clean_full_500":"Clean full training: 500 steps",
    "weight_clean_full_2500":"Clean full training: 2,500 steps", "output_resize_only":"Resize-only control",
    "output_nfpa_0":"NFPA delta 0 (regeneration control)", "output_nfpa_20":"NFPA delta 20", "output_nfpa_40":"NFPA delta 40 (primary)",
}


def pct(value):
    return "not measured" if value is None else f"{100*value:.1f}%"


def number(value):
    return "not measured" if value is None else f"{value:.4f}"


def interval(value):
    return "["+", ".join(pct(x) for x in value)+"]"


def binomial_ci(k,n):
    from scipy.stats import beta
    if not 0<=k<=n or n<1:raise ValueError("Invalid binomial counts")
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def slug(value):
    return re.sub(r"[^a-z0-9]+","-",value.lower()).strip("-")


class Publication:
    def __init__(self, output):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.cache={};self.sources={};self.galleries=[];self.examples={}
        self.snapshot={"schema_version":1,"pairs_per_gallery":8,"image_indices":sample_positions(512),
                       "text_positions":sample_positions(256),"text":{},"images":{},"historical":{}}

    def read(self,path):
        path=Path(path)
        if path not in self.cache:
            self.cache[path]=json.loads(path.read_text())
            self.sources[str(path.relative_to(ROOT))]={"sha256":sha(path),"bytes":path.stat().st_size}
        return self.cache[path]

    def write(self,name,text):
        path=self.output/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text.rstrip()+"\n")

    def json(self,name,value):
        self.write(name,json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False))

    def rows(self,path):
        value=self.read(path)
        return value if isinstance(value,list) else value["rows"]

    def text_gallery(self,key,title,original,candidate,before_scores,after_scores,threshold,question_map,
                     similarities=None,qualities=None,positions=None,notes="",gamma=False):
        before_ids=[r["id"] for r in original]
        candidate_map={r["id"]:r for r in candidate}
        if len(candidate_map)!=len(candidate):raise ValueError("Duplicate qualitative text IDs")
        positions=sample_positions(len(original)) if positions is None else positions
        pairs=[];records=[];summary=[]
        for position in positions:
            a=original[position];b=candidate_map[a["id"]];identity=a["id"]
            first=before_scores.get(identity);second=after_scores.get(identity)
            def label(score):
                if score is None:return "Owner score not measured"
                detected=score["score"]>=threshold if gamma else score["score"]>threshold
                value=f"p={score['p_value']:.2e}" if "p_value" in score else f"score={score['score']:.4f}"
                return ("Detected" if detected else "Not detected")+" | "+value
            extra=""
            sim=(similarities or {}).get(identity)
            if sim is not None:extra=f" | Jina={sim:.3f}"
            quality=(qualities or {}).get(identity,{})
            pair={"before":a["text"],"after":b["text"],"before_label":label(first),"after_label":label(second)+extra}
            pairs.append(pair)
            record={"id":identity,"position":position,"question":question_map.get(identity,a.get("prompt","")),
                    "before":a["text"],"after":b["text"],"before_score":first,"after_score":second,
                    "jina_cosine":sim,"quality":quality}
            records.append(record)
            summary.append({"Example":len(records),"Test position":position,"Before":label(first),"After":label(second),"Jina":number(sim)})
        asset=f"assets/{key}.webp";page=f"qualitative/{key}.md"
        text_grid(self.output/asset,title,pairs)
        lines=[f"# {title}","",notes,"", "Eight pairs selected by fixed test positions before inspecting detection or quality outcomes. Undetected originals and unsuccessful attacks remain included.","",
               f"![Sixteen-panel text preview](../{asset})","", "The figure previews up to the first 360 original characters per response; full texts below are unabridged. Emoji use Unicode codepoints in the preview to avoid missing-glyph boxes. Owner scores and quality metrics use full responses.","",table(summary,list(summary[0])),""]
        for i,record in enumerate(records,1):
            lines += [f"## Example {i}: test position {record['position']}","",f"Prompt ID: `{record['id']}`.","", "**Question / prompt**", "",full_text(record["question"]),"", "**Original watermarked-model output**", "",full_text(record["before"]),"", "**After / alternative output**", "",full_text(record["after"]),""]
            if record["quality"]:
                q=record["quality"]
                details={k:q[k] for k in ("nli_label","reverse_nli_label","nli_truncated","numeric_set_f1","mistral_perplexity") if k in q}
                lines += ["Recorded diagnostics: `"+json.dumps(details,ensure_ascii=True)+"`.",""]
        lines += ["[Text metrics](../text.md) | [Report overview](../README.md)"]
        self.write(page,"\n".join(lines));self.examples[key]=records
        self.galleries.append({"key":key,"kind":"text","page":page,"asset":asset,"sample_ids":[r["id"] for r in records],"panels":16})
        return page

    def image_gallery(self,key,title,records,threshold,notes):
        pairs=[];selected=[];summaries=[]
        for index in sample_positions(512):
            record=records[index]
            for side in ("before","after"):
                path=record[side]
                if isinstance(path,(str,Path)) and sha(path)!=record[side+"_sha256"]:
                    raise ValueError("Qualitative PNG differs from its recorded input")
            def label(side):
                score=record.get(side+"_bit_accuracy")
                limit=record.get(side+"_threshold",threshold)
                return f"class {record['label']} | "+("not scored" if score is None else f"bits={score:.3f} | "+("detected" if score>limit else "not detected"))
            pairs.append({"index":index,"before":record["before"],"after":record["after"],"before_label":label("before"),"after_label":label("after")})
            public={k:v for k,v in record.items() if k not in ("before","after")}
            public.update(index=index,before_label=label("before"),after_label=label("after"))
            selected.append(public)
            summaries.append({"Index":index,"Class":record["label"],"Before":label("before"),"After":label("after"),
                              "LPIPS":number(record.get("lpips")),"Class CLIP":number(record.get("clip_alignment"))})
        asset=f"assets/{key}.webp";page=f"qualitative/{key}.md"
        image_grid(self.output/asset,title,pairs)
        self.write(page,"\n".join([f"# {title}","",notes,"", "Eight pairs use fixed original indices 0, 64, 128, 192, 256, 320, 384 and 448. No detection or visual-quality filtering is applied.","",
            f"![Eight original/after image pairs](../{asset})","", "The lossless figure contains 16 panels. Each panel is displayed at 256 pixels; metrics use original native-resolution PNGs. Labels report each output's detector result, not guaranteed removal.","",
            table(summaries,list(summaries[0])),"", "[Image metrics](../images.md) | [Report overview](../README.md)"]))
        self.examples[key]=selected
        self.galleries.append({"key":key,"kind":"image","page":page,"asset":asset,"sample_ids":[r["index"] for r in selected],"panels":16})
        return page

    def text(self):
        manifest=self.read(TEXT/"manifest.json")
        self.snapshot["models"]={"target":{k:manifest["generator"][k] for k in ("name","revision")},
             "rewriter":{"name":"meta-llama/Meta-Llama-3.1-8B-Instruct","revision":manifest["models"]["meta-llama/Meta-Llama-3.1-8B-Instruct"]["revision"]},
             "reference":{k:manifest["reference"][k] for k in ("name","revision")}}
        partitions=self.read(TEXT/"partitions.json")["text"]
        questions={r["id"]:r["question"] for r in partitions["test"]}
        lines=["# Text watermarking and removal","", "Target: **Qwen/Qwen3.5-4B instruct**. BIRA and ordinary rewriting use **Llama-3.1-8B-Instruct**. Every current test arm has 256 responses. TextSeal 1 and deleted Qwen-Base experiments are excluded.","",
            "TextSeal 2 changes the sampler. The DistSeal-inspired text adaptation learns a hidden-state teacher and distils it into merged student weights; this is not an official DistSeal text implementation.","",
            "## Access and interpretation","", "BIRA and ordinary paraphrasing receive text only. Direct resampling and smoothing receive simulated top-10 model probabilities. For TextSeal 2 these are **pre-watermark underlying logits**: results do not establish bypass when an API exposes only its keyed sampling distribution. For the student they come from the merged watermarked model. Smoothing uses an independent Qwen3.5-0.8B instruct reference.","",
            "Direct resampling and smoothing generate fresh answers to the same question; their similarity to an earlier answer is not an entailment requirement. Jina cosine is continuous similarity, not a preservation percentage. NLI is reported only for rewrites, using entailment in both directions.",""]
        gallery_rows=[]
        for case,title in (("textseal2","TextSeal 2"),("latent","DistSeal-inspired student")):
            root=TEXT/case;d=self.read(root/"detection.json");q=self.read(root/"quality.json")
            jina=self.read(JINA/case/"summary.json")
            self.snapshot["models"]["jina_v5"]={k:jina["model"][k] for k in ("name","revision")}
            self.snapshot["text"][case]={"detection":d,"quality":q,"jina":{k:jina[k] for k in ("conditions","paired_deltas_vs_vanilla","versions")}}
            lines += [f"## {title}",""]
            if case=="textseal2":
                lines += ["The released v2 deduplicated Gamma detector uses nominal p-value thresholds 0.001 (primary) and 0.01. The 256-example null sets cannot precisely validate a 0.1% false-positive rate.",""]
            else:
                lines += [f"The primary threshold is the frozen validation-derived score **>{d['thresholds']['fpr001']:.10f}**, with a secondary threshold **>{d['thresholds']['fpr01']:.10f}**. Labels `fpr001`/`fpr01` are target operating points, not measured guarantees or Gamma p-values.",""]
            rows=[]
            for name,metrics in d["conditions"].items():
                first=metrics["operating_points"]["fpr001"];second=metrics["operating_points"]["fpr01"]
                label=METHODS.get(name,{"original":"Original marked output","clean":"Clean-model negative control","teacher":"External learned teacher"}.get(name,name))
                rows.append({"Condition":label,"n":metrics["n"],"Detected primary":f"{first['detected']}/{metrics['n']} ({pct(first['detection_rate'])})",
                             "Primary 95% CI":interval(binomial_ci(first["detected"],metrics["n"])),
                             "Detected secondary":pct(second["detection_rate"]),"Eligible":metrics["eligible"],"AUROC vs human":number(metrics["auroc"])})
            lines += [table(rows,list(rows[0])),"", "Observed human false positives: "+str(d["conditions"]["original"]["operating_points"]["fpr001"]["human_false_positives"])+"/256 at the primary threshold. The clean-model row reports the separate model-output false-positive rate.","", "### Quality and similarity",""]
            rows=[]
            for name in METHODS:
                value=q[name];sim=jina["conditions"][name]
                rows.append({"Method":METHODS[name],"Jina cosine":number(sim["mean"]),"Jina mean 95% CI":"["+", ".join(f"{x:.4f}" for x in sim["mean_ci95"])+"]",
                             "Bidirectional NLI":pct(value.get("bidirectional_entailment_fraction")) if name in ("vanilla","bira") else "N/A: fresh answer",
                             "Mistral PPL":number(value.get("mistral_perplexity")),"Numeric-set F1":number(value.get("numeric_set_f1")) if name in ("vanilla","bira") else "N/A: fresh answer"})
            lines += [table(rows,list(rows[0])),"", "PPL uses the independent Mistral-7B evaluator; lower is more probable under that model, not proof of truth or utility. Numeric-set F1 checks sets of numbers rather than their roles.","", "### Output diagnostics",""]
            rows=[]
            originals=self.rows(root/"inputs.json")
            caps=sum(r.get("length_capped",False) for r in originals)
            lines += [f"Original generation budget: 200 Qwen tokens; **{caps}/256 originals hit the budget**. These possibly unfinished outputs are retained rather than filtered.",""]
            for name,value in q.items():
                cap=value.get("length_capped")
                if cap is None:
                    generated=self.rows(root/(name+".json"))
                    if all("tokens" in row for row in generated):cap=sum(len(row["tokens"])>=200 for row in generated)
                rows.append({"Method":METHODS[name],"Empty":value["empty"],"Flagged degeneration":value.get("degenerated","not evaluated"),
                             "Length cap":cap if cap is not None else "not recorded","Mean attempts":number(value.get("mean_attempts")),
                             "NLI truncated":f"{round(value['nli_truncated']*value['n'])}/{value['n']}" if "nli_truncated" in value else "N/A",
                             "Trigram repetition":number(value.get("trigram_repetition"))})
            self.snapshot["text"][case]["original_budget_caps"]=caps
            self.snapshot["text"][case]["output_diagnostics"]=rows
            lines += [table(rows,list(rows[0])),"", "Missing diagnostics are not treated as zero. The NLI model truncates overlong pairs to its 512-token context. Full per-condition summaries, both thresholds and all similarity controls are in the [portable metric snapshot](data/metrics.json).",""]
            original=self.rows(root/"inputs.json");before={r["id"]:r for r in self.rows(root/"original_scores.json")}
            transitions=[]
            for method in METHODS:
                after={r["id"]:r for r in self.rows(root/(method+"_scores.json"))}
                limit=d["thresholds"]["fpr001"]
                decision=lambda record:record["score"]>=limit if case=="textseal2" else record["score"]>limit
                positives=[identity for identity,row in before.items() if decision(row)]
                became_negative=sum(not decision(after[identity]) for identity in positives)
                became_positive=sum(decision(after[identity]) for identity,row in before.items() if not decision(row))
                transitions.append({"Method":METHODS[method],"Originally detected":len(positives),"Now undetected":became_negative,
                                    "Fraction of original positives":pct(became_negative/len(positives)),"Original negatives now detected":became_positive})
                similarities={r["id"]:r["cosine"] for r in self.rows(JINA/case/(method+"_similarity.json"))}
                qualities={r["id"]:r for r in self.rows(root/(method+"_quality.json"))}
                path=self.text_gallery(f"text-{case}-{method}",title+": "+METHODS[method],original,self.rows(root/(method+".json")),before,after,d["thresholds"]["fpr001"],questions,
                    similarities,qualities,notes="This is a fresh answer to the same prompt, not a rewrite." if method in ("direct_topk","smoothing") else "The candidate is a rewrite of the original response. Both detection and preservation failures are retained.",gamma=case=="textseal2")
                gallery_rows.append({"Watermark":title,"Method":METHODS[method],"Eight pairs":f"[View]({path})"})
            self.snapshot["text"][case]["paired_transitions_primary"]=transitions
            lines += ["### Paired detector transitions","",table(transitions,list(transitions[0])),"",
                "These counts show how many initially detected outputs become undetected. They exclude pre-existing false negatives from the denominator. For resampling/smoothing, this is a transition between fresh responses to the same prompt, not necessarily removal from the original text.",""]
        latent=self.read(TEXT/"latent_manifest.json");training=self.read(TEXT/"latent_training/latent_embedder/results.json")
        self.snapshot["text"]["training"]={"teacher_steps":training["steps"],"student_steps":latent["text_training"]["steps"],
             "lora_rank":latent["text_training"]["lora_rank"],"best_teacher_validation_bit_accuracy":training["best_validation_bit_accuracy"]}
        lines += ["## Watermark creation and student training","",table([
            {"Stage":"TextSeal 2","Training":"No weight training","Sampler":"temperature 1, top-p .95, cap 200; rho .9, ngram 3","Measured outcome":"253/256 original responses detected at p<=.001"},
            {"Stage":"Latent teacher","Training":f"{training['steps']:,} steps","Sampler":"learned hidden-state hook","Measured outcome":f"Best validation bit accuracy {pct(training['best_validation_bit_accuracy'])}; not per-response detection"},
            {"Stage":"Distilled student","Training":f"{latent['text_training']['steps']:,} steps, LoRA rank {latent['text_training']['lora_rank']}","Sampler":"ordinary sampling from merged weights","Measured outcome":"118/256 original student responses detected at primary threshold"},
        ],["Stage","Training","Sampler","Measured outcome"]),"", "## Qualitative comparisons","",table(gallery_rows,list(gallery_rows[0])),"", "All eight pairs per method use the same fixed positions; figures are previews and their pages retain full verbatim text. [Historical paper controls](historical.md) are separate from these current target experiments.","", "[Overview](README.md) | [Metric definitions](metrics.md)"]
        self.write("text.md","\n".join(lines))

    def images(self):
        lines=["# Image watermarking and removal","", "Targets are released DistSeal **RAR-XL** (256 pixels) and **UViT-H/DC-AE** (512 pixels). All current conditions have 512 marked test outputs, 512 clean test controls and 256 clean calibration images. Owner detectors evaluate frozen outputs after attacks.","",
            "## Access settings","",table([
                {"Method":"Merging / clean replacement","Target weights":"Yes","Original clean weights":"Yes","Other access":"No detector feedback"},
                {"Method":"Clean LoRA / full updates","Target weights":"Yes","Original clean weights":"No","Other access":"Independent clean fitting images"},
                {"Method":"NFPA","Target weights":"No","Original clean weights":"No","Other access":"Output PNGs plus independent public SD2.1 model"},
                {"Method":"JPEG / center crop","Target weights":"No","Original clean weights":"No","Other access":"Output images only"},
            ],["Method","Target weights","Original clean weights","Other access"]),"",
            "NFPA is the authors' [NeurIPS 2025 method](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b48405307576689bea14d221ccba73ae-Abstract-Conference.html), using an independent SD2.1 public mirror. The original SD2.1 host returned 404; byte-for-byte equality with that unavailable host is not claimed. This is output purification using preserved PNGs to simulate a closed provider, not a live API or modification of the provider's weights.","",
            "All quality metrics use the same 512 reference PNGs, verified bit-identical across source experiments. High paired distances after a weight change can reflect a different valid generated sample; they are not automatically degraded generation quality. Class-text CLIP complements image similarity.",""]
        gallery_rows=[];robust_rows=[]
        nfpa_manifest=self.read(NFPA/"manifest.json")
        public=self.read(nfpa_manifest["cases"][0]["task"])["public_model"]
        self.snapshot["models"]["image_metrics"]={"CLIP":{"name":"openai/clip-vit-base-patch32","revision":nfpa_manifest["clip"]["revision"]},
                                                    "NFPA_public_model":{k:public[k] for k in ("name","revision")}}
        for family,weight_case,nfpa_case,title in (("rar","model02","model01","RAR-XL"),("dcae","model03","model02","UViT-H/DC-AE")):
            quality=self.read(QUALITY/family/"results.json")
            pools=self.read(QUALITY/family/"pairs.json")
            old=self.read(WEIGHTS/"evaluation"/weight_case/"results.json")
            regen=self.read(NFPA/nfpa_case/"results.json")
            original={r["index"]:r for r in self.rows(IMAGES/family/"watermarked/samples.json")}
            self.snapshot["images"][family]={"quality":quality,"weight_detection":old,"nfpa_detection":regen}
            lines += [f"## {title}","",f"Original frozen owner threshold: **>{old['threshold']}**. Higher bit agreement indicates watermark detection. Every primary row uses this same threshold.",""]
            detection_rows=[];quality_rows=[];recalibration=[]
            for name,measured in quality["conditions"].items():
                label=IMAGE_LABELS[name];pool=pools[name]
                if name.startswith("weight_"):
                    condition=name[len("weight_"):];raw=old["conditions"][condition]["all"]
                    candidates={r["index"]:r for r in self.rows(WEIGHTS/"evaluation"/weight_case/condition/"samples.json")}
                else:
                    condition=name[len("output_"):];raw=regen["conditions"][condition]["all"]["original_threshold"]
                    owner_scores={r["id"]:r for r in self.read(NFPA/nfpa_case/(condition+"_scores.json"))}
                    candidates={r["index"]:owner_scores[Path(r["candidate"]).stem] for r in pool["rows"]}
                    recal=regen["conditions"][condition]["all"]["transformed_calibration_threshold"]
                    recalibration.append({"Condition":label,"Recalibrated threshold":number(recal["threshold"]),"Detected":pct(recal["detection"]["rate"]),
                                          "Clean FPR":pct(recal["false_positive"]["rate"]),"AUROC":number(recal["auroc"])})
                det=raw["detection"];false=raw["false_positive"]
                detection_rows.append({"Method":label,"n":det["n"],"Detected":pct(det["rate"]),"Detection 95% CI":interval(det["ci95"]),
                                       "Clean FPR":pct(false["rate"]),"AUROC":number(raw["auroc"])})
                q=measured["quality"]["all"]
                quality_rows.append({"Method":label,"LPIPS ↓":number(q["lpips"]["mean"]),"SSIM ↑":number(q["ssim"]["mean"]),
                                     "PSNR dB ↑":number(q["psnr"]["mean"]),"Image CLIP ↑":number(q["clip_image_cosine"]["mean"]),"Class CLIP ↑":number(q["clip_alignment"]["mean"])})
                per_image={r["index"]:r for r in self.rows(QUALITY/family/(name+".json"))}
                records={r["index"]:{"index":r["index"],"label":r["label"],"before":r["reference"],"after":r["candidate"],
                      "before_sha256":r["reference_sha256"],"after_sha256":r["candidate_sha256"],
                      "before_bit_accuracy":original[r["index"]]["bit_accuracy"],"after_bit_accuracy":candidates[r["index"]]["bit_accuracy"],
                      "lpips":per_image[r["index"]]["lpips"],"clip_alignment":per_image[r["index"]]["clip_alignment"]} for r in pool["rows"]}
                note="Candidate images are generated by modified weights using the same requested class/index/seed. They can be different valid samples."
                if name.startswith("output_"):
                    note="Candidate images are transformations of the original PNG. NFPA uses only the image and an independent public model. Delta 0 still executes VAE/DDIM and zero-flow resampling."
                page=self.image_gallery(f"image-{family}-{slug(name)}",title+": "+label,records,old["threshold"],note)
                gallery_rows.append({"Model":title,"Method":label,"Eight pairs":f"[View]({page})"})
            lines += ["### Detection at the fixed original threshold","",table(detection_rows,list(detection_rows[0])),"", "### Matched quality metrics","",table(quality_rows,list(quality_rows[0])),"",
                      "Means are over the same 512 pairs. Per-condition quantiles and 95% bootstrap intervals, plus class-zero and unseen-class groups, are retained in the [portable metric snapshot](data/metrics.json). PSNR identity values are capped at 120 dB.","",
                      "### NFPA with separately transformed calibration","",table(recalibration,list(recalibration[0])),"", "These owner-side recalibrations use separate attacked clean calibration images; they do not select the attack strength. Original-threshold results above remain the primary comparison.",""]
            for condition,groups in old["conditions"].items():
                for transform,raw in groups["all"].get("attacks",{}).items():
                    robust_rows.append({"Model":title,"Weight condition":IMAGE_LABELS["weight_"+condition],"Output transform":transform,"n":raw["detection"]["n"],
                                        "Detected":pct(raw["detection"]["rate"]),"Clean FPR":pct(raw["false_positive"]["rate"]),"AUROC":number(raw["auroc"]),"Threshold":number(raw["threshold"])})
            for transform in ("jpeg50","center_crop_area50"):
                raw=old["conditions"]["unmodified_supplied"]["all"]["attacks"][transform]
                records={}
                for index in sample_positions(512):
                    before=IMAGES/family/"watermarked"/f"{index:05}.png"
                    with Image.open(before) as pic:
                        picture=pic.convert("RGB")
                    if transform=="jpeg50":
                        buffer=BytesIO();picture.save(buffer,format="JPEG",quality=50);buffer.seek(0)
                        with Image.open(buffer) as jpeg:after=jpeg.convert("RGB")
                    else:
                        side=int(picture.width*.5**.5);left=round((picture.width-side)/2);top=round((picture.height-side)/2)
                        after=picture.crop((left,top,left+side,top+side))
                    row=original[index]
                    records[index]={"index":index,"label":row["label"],"before":before,"after":after,
                         "before_sha256":sha(before),"after_pixel_sha256":__import__("hashlib").sha256(after.tobytes()).hexdigest(),
                         "before_bit_accuracy":row["bit_accuracy"],"after_bit_accuracy":row["attacks"][transform],
                         "before_threshold":old["threshold"],"after_threshold":raw["threshold"],"reconstructed_display_transform":transform}
                page=self.image_gallery(f"image-{family}-{transform}",title+": "+transform,records,old["threshold"],
                    "This reconstructs the documented JPEG-quality-50 or native center-crop operation from the saved original PNG for display. Captions use the originally recorded detector scores and transformed-calibration threshold. Paired quality was not measured for these legacy output transformations.")
                gallery_rows.append({"Model":title,"Method":transform+" on original marked output","Eight pairs":f"[View]({page})"})
        lines += ["## Qualitative comparisons","",table(gallery_rows,list(gallery_rows[0])),"", "[Historical image pilots](historical.md) have their own galleries and are not pooled with the current eight-thousand-image fitting protocol.","", "[JPEG/crop robustness tables](robustness.md) | [Overview](README.md) | [Metric definitions](metrics.md)"]
        self.write("images.md","\n".join(lines))
        self.robust_rows=robust_rows

    def historical(self):
        lines=["# Historical and incomplete watermark experiments","", "These retained runs are separate protocols. They are not pooled with the current instruct-Qwen and generic clean-training comparisons. Retired TextSeal 1 and deleted Qwen-Base outputs are excluded entirely.","",
               "## Earlier image merging and single-class clean fine-tuning","", "The original image run used the released merge implementation and a clean LoRA pilot on 50 images of class zero. The later primary weight-removal run uses an explicitly separated 8,000-image, 800-class fitting protocol. Different recipes can give different results even for similarly named attacks.",""]
        gallery_rows=[]
        for family,title in (("rar","RAR-XL"),("dcae","UViT-H/DC-AE")):
            baseline=self.read(IMAGES/family/"results.json")
            originals={r["index"]:r for r in self.rows(IMAGES/family/"watermarked/samples.json")}
            self.snapshot["historical"][family]={"baseline":baseline}
            controls=[]
            for condition,groups in baseline["conditions"].items():
                raw=groups["all"]
                controls.append({"Control":condition,"n":raw["detection"]["n"],"Detected":pct(raw["detection"]["rate"]),
                                 "Clean FPR":pct(raw["false_positive"]["rate"]),"AUROC":number(raw["auroc"]),
                                 "Bit accuracy":number(raw["mean_bit_accuracy"]),"Class CLIP":number(raw["clip_alignment"])})
                for transform,value in raw.get("attacks",{}).items():
                    self.robust_rows.append({"Model":title+" (original control)","Weight condition":condition,"Output transform":transform,"n":value["detection"]["n"],
                        "Detected":pct(value["detection"]["rate"]),"Clean FPR":pct(value["false_positive"]["rate"]),"AUROC":number(value["auroc"]),"Threshold":number(value["threshold"])})
            lines += ["### "+title+": original generation controls","",table(controls,list(controls[0])),""]
            rows=[]
            for branch in ("base","clean"):
                root=IMAGES/(family+"_"+branch);data=self.read(root/"results.json")
                self.snapshot["historical"][family][branch]=data
                for condition,groups in data["conditions"].items():
                    raw=groups["all"];candidate={r["index"]:r for r in self.rows(root/condition/"samples.json")}
                    label=(f"Historical merge: {100*float(condition.rsplit('_',1)[-1]):g}% watermarked") if branch=="base" else ("50-image clean LoRA "+condition.rsplit("_",1)[-1]+" steps")
                    rows.append({"Method":label,"n":raw["detection"]["n"],"Detected":pct(raw["detection"]["rate"]),"Clean FPR":pct(raw["false_positive"]["rate"]),
                                 "AUROC":number(raw["auroc"]),"Class CLIP":number(raw["clip_alignment"]),"Paired LPIPS/SSIM/PSNR":"not measured"})
                    records={i:{"index":i,"label":originals[i]["label"],"before":IMAGES/family/"watermarked"/f"{i:05}.png",
                        "after":root/condition/f"{i:05}.png","before_sha256":sha(IMAGES/family/"watermarked"/f"{i:05}.png"),
                        "after_sha256":sha(root/condition/f"{i:05}.png"),"before_bit_accuracy":originals[i]["bit_accuracy"],
                        "after_bit_accuracy":candidate[i]["bit_accuracy"],"clip_alignment":candidate[i]["clip_alignment"]} for i in sample_positions(512)}
                    page=self.image_gallery(f"legacy-{family}-{branch}-{slug(condition)}",title+": "+label,records,data["threshold"],
                        "Historical protocol. Class alignment and detection were recorded; paired perceptual quality metrics were not measured. The same fixed indices are shown without success filtering.")
                    gallery_rows.append({"Model":title,"Historical method":label,"Eight pairs":f"[View]({page})"})
                    for transform,value in raw.get("attacks",{}).items():
                        self.robust_rows.append({"Model":title+" (historical)","Weight condition":label,"Output transform":transform,"n":value["detection"]["n"],
                            "Detected":pct(value["detection"]["rate"]),"Clean FPR":pct(value["false_positive"]["rate"]),"AUROC":number(value["auroc"]),"Threshold":number(value["threshold"])})
            lines += ["### "+title,"",table(rows,list(rows[0])),""]
        lines += ["### Historical image galleries","",table(gallery_rows,list(gallery_rows[0])),"", "## Independent OPT/Llama paper controls","",
             "These use OPT-1.3B as the generator and Llama-3.1-8B-Instruct for rewriting. They are historical paper-scheme controls, not TextSeal 2/Qwen results. Only the recorded KGW detection evaluation is complete; quality and the other schemes are incomplete or absent.",""]
        paper_inventory=[]
        for run in sorted(PAPER.iterdir()):
            provenance=self.read(run/"provenance.json")
            for case in provenance["cases"]:
                folder=run/case["name"];counts={}
                for arm,file in (("original","generated.json"),("vanilla","vanilla.json"),("bira","bira.json")):
                    counts[arm]=len(self.rows(folder/file)) if (folder/file).exists() else 0
                paper_inventory.append({"Run":run.name,"Scheme":case["scheme"],"Original outputs":counts["original"],"Ordinary rewrites":counts["vanilla"],"BIRA rewrites":counts["bira"],
                    "Owner metrics":"recorded" if (folder/"detection.json").exists() else "not measured"})
                if (folder/"detection.json").exists():
                    detection=self.read(folder/"detection.json")
                    self.snapshot["historical"]["paper_"+run.name+"_"+case["scheme"]]={"detection":detection}
                    measured=[]
                    self.snapshot["historical"]["paper_"+run.name+"_"+case["scheme"]]["partial_quality"]={}
                    for arm,value in detection["conditions"].items():
                        quality_path=folder/(arm+"_quality.json")
                        quality=self.rows(quality_path) if quality_path.exists() else []
                        nli=[r for r in quality if "nli_label" in r and "reverse_nli_label" in r]
                        partial={"observed_n":len(quality),"target_n":value["n"],"bidirectional_nli_n":len(nli),
                                 "bidirectional_nli_passes":sum(r["nli_label"]==r["reverse_nli_label"]=="entailment" for r in nli)}
                        self.snapshot["historical"]["paper_"+run.name+"_"+case["scheme"]]["partial_quality"][arm]=partial
                        measured.append({"Condition":arm,"n":value["n"],"Detected":pct(value["detection_rate"]),"Human FPR":pct(value["human_false_positives"]/value["human_n"]),"AUROC":number(value["auroc"]),
                            "Quality coverage":f"{len(quality)}/{value['n']}","Bidirectional NLI":pct(statistics.fmean(r["nli_label"]==r["reverse_nli_label"]=="entailment" for r in nli)) if nli else "not measured"})
                    lines += [f"### {case['scheme']}: recorded detection","",f"Frozen C4-validation threshold: >{detection['threshold']:.10f}. Quality measurements below refer only to their stated partial coverage.","",table(measured,list(measured[0])),""]
                if counts["original"]>=8 and min(counts["vanilla"],counts["bira"])>=8:
                    original=self.rows(folder/"generated.json");common=min(counts.values());positions=sample_positions(common)
                    first_path=folder/"original_scores.json";before={r["id"]:r for r in self.rows(first_path)} if first_path.exists() else {}
                    for arm in ("vanilla","bira"):
                        score_path=folder/(arm+"_scores.json");after={r["id"]:r for r in self.rows(score_path)} if score_path.exists() else {}
                        limit=self.read(folder/"detection.json")["threshold"] if (folder/"detection.json").exists() else 0.
                        page=self.text_gallery(f"paper-{case['scheme'].lower()}-{arm}","Historical "+case["scheme"]+": "+METHODS[arm],original,self.rows(folder/(arm+".json")),before,after,limit,{},
                            positions=positions,notes=f"Historical OPT/Llama control. Fixed positions are drawn from the first {common} responses available in all arms. Missing owner or quality scores remain explicitly unmeasured.")
                        lines += [f"[Eight {case['scheme']} / {METHODS[arm]} pairs]({page}).",""]
        self.snapshot["historical"]["paper_inventory"]=paper_inventory
        lines += ["### Retained artifact coverage","",table(paper_inventory,list(paper_inventory[0])),"", "Zero retained outputs means no completed measurements are available here. The earlier 60/163-output staging run is superseded by the later retained run, not an additional completed benchmark. Stale receipts do not turn partial outputs into complete quality evaluations.","", "[Overview](README.md)"]
        self.write("historical.md","\n".join(lines))
        self.write("robustness.md","\n".join(["# Recorded JPEG and crop robustness","", "These are all recorded JPEG-quality-50 and center-crop-area-50 detector measurements for the retained weight-update and historical image conditions. Each transform uses its separately frozen clean-calibration threshold. These rows do not measure paired perceptual quality; those cells are not silently filled from untransformed images.","",table(self.robust_rows,list(self.robust_rows[0])),"", "[Image metrics and transform galleries](images.md) | [Historical protocols](historical.md) | [Overview](README.md)"]))

    def overview(self):
        self.snapshot["datasets"]={"current_text":"HC3 questions split by normalized question hash; disjoint train/validation/test/removal groups",
                                   "images":"Saved class-conditional synthetic ImageNet outputs; no real fitting images exported",
                                   "historical_paper_controls":"Independent C4 prompt groups"}
        self.write("metrics.md","""# Metric definitions and limits

| Metric | Meaning | Direction / limit |
|---|---|---|
| Detection rate | Fraction of marked outputs above the frozen owner threshold | Lower means more detector evasion; not necessarily useful removal |
| False-positive rate | Fraction of independent clean/human outputs above that threshold | Lower is better; finite samples cannot prove extreme FPRs |
| AUROC | Ranking marked versus negative outputs | 0.5 indicates chance ranking on this sample |
| Bit accuracy | Fraction of registered watermark bits recovered | Different from per-output detection rate |
| Jina v5 cosine | Original/candidate text embedding cosine, symmetric text-matching task | Continuous similarity; not a preservation percentage |
| Bidirectional NLI | Original entails rewrite and rewrite entails original, both argmax labels | Automated equivalence proxy; 512-token truncation can affect it |
| Numeric-set F1 | Set overlap of numbers between original and rewrite | Does not validate which number refers to which fact |
| Mistral perplexity | Token likelihood under independent Mistral-7B | Lower is more probable, not necessarily true or useful |
| LPIPS-Alex v0.1 | Perceptual distance between paired images | Lower means closer; not a percentage |
| SSIM | Native-resolution structural similarity | Higher means closer; sensitive to alignment/viewpoint |
| PSNR | Native-resolution pixel error on unit-range images | Higher means closer; identity capped at 120 dB |
| Image-image CLIP | Cosine between paired image embeddings | Broad semantic similarity, not identical details or identity |
| Class-text CLIP | Cosine with `a photo of a <ImageNet class>` | Agreement with requested class, not comprehensive visual quality |

Text tables derive exact 95% Clopper-Pearson binomial intervals from the saved
counts. Image binomial intervals are those in the original owner evaluation.
Jina and matched image-quality mean intervals are paired-example bootstrap
intervals from the saved analyses (2,000 resamples). Intervals describe these
fixed data, not a guarantee across new domains, languages or generators.

The text watermark thresholds differ: TextSeal 2 uses a Gamma p-value test,
whereas the student uses a learned extractor score with thresholds fitted on
disjoint validation outputs. A nominal or target FPR label is not an observed
false-positive rate. Primary and secondary thresholds are both shown.

Image weight changes can generate another valid sample even for the same class
and seed. Paired metrics therefore measure sample drift and must be considered
alongside class alignment. NFPA modifies a particular input image, so its access
and preservation setting differs. Neither setting has a universal quality floor
or human-verified preservation pass rate in this repository.

Qualitative examples use eight fixed positions, not best-performing outputs.
Their owner decisions and failure cases remain visible. Image grids use lossless
display encoding after uniform resizing; quantitative metrics always use native
images. Text figures are labeled previews, with full verbatim responses and
questions on the linked pages. Public snapshots omit watermark secrets, payloads,
checkpoints and machine-specific paths.

[Overview](README.md)
""")
        text_rows=[]
        for case,title in (("textseal2","TextSeal 2"),("latent","DistSeal-inspired student")):
            d=self.snapshot["text"][case]["detection"]["conditions"]
            for name in ("original",*METHODS):
                x=d[name]["operating_points"]["fpr001"]
                text_rows.append({"Watermark":title,"Condition":"Original marked output" if name=="original" else METHODS[name],
                                  "Detected":pct(x["detection_rate"]),"n":d[name]["n"]})
        image_rows=[]
        for family,title in (("rar","RAR-XL"),("dcae","UViT-H/DC-AE")):
            data=self.snapshot["images"][family]["quality"]["conditions"]
            for name in ("weight_unmodified_supplied","weight_clean_lora_2500","weight_clean_full_2500","output_nfpa_40"):
                x=data[name];q=x["quality"]["all"]
                image_rows.append({"Model":title,"Condition":IMAGE_LABELS[name],"Detected":pct(x["detection"]["rate"]),
                                   "LPIPS":number(q["lpips"]["mean"]),"Class CLIP":number(q["clip_alignment"]["mean"])})
        self.write("README.md","\n".join(["# Watermarking and removal: retained results","",
            "This is the portable publication report for the completed watermarking experiments. All tables, selected example texts and comparison figures are tracked under this directory and render without the ignored local results, datasets or model caches.","",
            "## Start here","",table([
                {"Page":"[Text metrics](text.md)","Coverage":"TextSeal 2 and DistSeal-inspired Qwen instruct teacher/student; all four removal methods and controls"},
                {"Page":"[Image metrics](images.md)","Coverage":"RAR and DC-AE; all weight merges, clean LoRA/full updates, NFPA strengths and controls"},
                {"Page":"[JPEG/crop robustness](robustness.md)","Coverage":"Every retained recorded image output-transform/weight-condition combination"},
                {"Page":"[Historical appendix](historical.md)","Coverage":"Earlier single-class image pilots and incomplete independent OPT/Llama paper controls"},
                {"Page":"[Metric definitions](metrics.md)","Coverage":"Thresholds, uncertainty, access assumptions and interpretation"},
                {"Page":"[Portable metrics](data/metrics.json)","Coverage":"Machine-readable full summaries, including both text operating points and image subgroups"},
                {"Page":"[Selected examples](data/examples.json)","Coverage":"Full sampled texts, image input hashes, owner scores and fixed sample IDs"},
                {"Page":"[Audit manifest](publication.json)","Coverage":"Hashes of the published files and source result identifiers"},
            ],["Page","Coverage"]),"",
            "Primary targets are Qwen/Qwen3.5-4B instruct, released DistSeal RAR-XL and UViT-H/DC-AE. BIRA uses Llama-3.1-8B-Instruct. DistSeal-inspired text is an experimental adaptation, not an official paper text implementation. TextSeal 1 and deleted Qwen-Base runs are excluded.","",
            "## Text snapshot","",table(text_rows,list(text_rows[0])),"",
            "TextSeal 2's primary criterion is nominal p<=0.001. The student uses a different frozen validation-derived extractor threshold; it is not a Gamma p-value. Raw evasion must be read alongside the NLI, similarity and quality tables.","",
            "## Image snapshot","",table(image_rows,list(image_rows[0])),"",
            "All primary image settings and quality metrics are in the full [image tables](images.md). Weight access and output-only access differ. Paired distances after a weight change can reflect a different valid image rather than poor generation quality.","",
            "## Qualitative examples: eight pairs each","", "Every method page provides a 16-panel comparison and the fixed sample IDs. Captions retain unsuccessful attacks and originally undetected text outputs. No examples were selected using detection or aesthetic quality.","",
            "### Example image grid: DC-AE / NFPA delta 40","", "![Eight watermarked/after image pairs](assets/image-dcae-output-nfpa-40.webp)","",
            "[Full comparison and per-pair scores](qualitative/image-dcae-output-nfpa-40.md). The figure can contain visible detail and framing changes; it does not assert exact semantic preservation.","",
            "### Example text grid: TextSeal 2 / BIRA","", "![Eight original/after text previews](assets/text-textseal2-bira.webp)","",
            "[Full verbatim responses, prompts and diagnostics](qualitative/text-textseal2-bira.md). The image is a preview, not the complete response.","",
            "## Sources and reproduction","", "The source experiments and revisions are identified in the [source index](data/sources.json). These identifiers describe local provenance; readers do not need those private filesystem locations to view this report. Raw model weights and full datasets remain outside Git.","",
            "With retained local experiment artifacts available, rebuild from the repository root:","", "```bash", "PYTHONPATH=. uv run --no-sync python scripts/publish_watermark_results.py", "PYTHONPATH=. uv run --no-sync python scripts/publish_watermark_results.py --check", "```", "",
            "The check command uses only published files, validates artifact hashes and local Markdown links, and verifies eight pairs per figure. Historical partial quality is explicitly labeled with its observed coverage; planned schemes without retained outputs are not presented as completed tests.","",
            "### Method references","", "- [TextSeal 2](https://arxiv.org/abs/2610.40613).", "- [DistSeal, ICML 2026](https://proceedings.mlr.press/v306/rebuffi26a.html); text here is an experimental adaptation.",
            "- [BIRA](https://arxiv.org/abs/2509.23019), using the released Llama implementation.",
            "- Confidence smoothing: *Watermark Smoothing Attacks against Language Models*, Findings EMNLP 2025; the unknown-watermark mixing variant here is an adaptation.",
            "- [NFPA, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b48405307576689bea14d221ccba73ae-Abstract-Conference.html).", "",
            "Method protocols: [text](../instruct_watermarking.md), [image weight access](../watermarking.md), [NFPA output-only](../closed_image_watermarking.md), [matched quality](../matched_image_quality.md), [Jina](../jina_similarity.md)."] ))

    def build(self):
        self.text();self.images();self.historical();self.overview()
        self.json("data/metrics.json",self.snapshot)
        self.json("data/examples.json",self.examples)
        self.json("data/sources.json",{"source_files":self.sources,"note":"Relative identifiers refer to retained local artifacts; no raw checkpoint or watermark secret is exported."})
        self.json("publication.json",{"schema_version":1,"selection":"Eight evenly spaced positions without detection/quality filtering; partial paper controls use the common completed prefix.",
            "builder_sha256":sha(Path(__file__)),"renderer_sha256":sha(Path(__file__).with_name("watermark_report_assets.py")),
            "renderer":{"Pillow":pillow_version,"font":"DejaVuSans","font_sha256":sha(font(14).path),"image_display_pixels":256,"encoding":"WebP lossless"},
            "galleries":self.galleries,"files":{str(p.relative_to(self.output)):sha(p) for p in sorted(self.output.rglob("*")) if p.is_file() and p.name!="publication.json"}})
        return validate_publication(self.output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"docs/watermarking")
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--legacy-markdown",action="store_true",help="Rebuild the historical Markdown export in a separate --output directory")
    args=parser.parse_args()
    if args.legacy_markdown and args.output.resolve()==(ROOT/"docs/watermarking").resolve():
        parser.error("Legacy Markdown exports require a separate --output directory; canonical research documentation is LaTeX")
    print(json.dumps(Publication(args.output).build() if args.legacy_markdown and not args.check else validate_publication(args.output),indent=2))
