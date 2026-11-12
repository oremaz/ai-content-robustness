"""Post-hoc preservation-qualified removal for the retained matched image pairs."""

import hashlib
import json
from pathlib import Path

from ai_content_detector.watermarking.preservation import joint_outcomes, numeric_gate


ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "results/research_workflow/image_quality/20261104T153348508006Z"
ORIGINAL = ROOT / "results/research_workflow/watermarking/20261026T001727478096Z"
NFPA = ROOT / "results/research_workflow/closed_image/20261104T082631753968Z"


def main():
    sources = {}

    def read(path):
        path = Path(path)
        data = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    def indexed(rows, field="index"):
        result = {r[field]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate paired sample identifiers")
        return result

    nfpa = read(NFPA / "manifest.json")
    results = []
    current_results = []
    current_rule_path = ROOT / "tex/provenance/revised_quality_rule_no_ssim.json"
    current_rule = read(current_rule_path)["quality_rule"]
    for family, case_name in [("rar", "model01"), ("dcae", "model02")]:
        originals = indexed(read(ORIGINAL / family / "watermarked/samples.json")["rows"])
        threshold = read(ORIGINAL / family / "results.json")["threshold"]
        original_decisions = {i: r["bit_accuracy"] > threshold for i, r in originals.items()}
        pools = read(QUALITY / family / "pairs.json")
        task = read(next(c["task"] for c in nfpa["cases"] if c["name"] == case_name))
        input_hashes = {r["id"]: r["sha256"] for r in task["images"]}
        reference_hashes = {r["reference_sha256"]: r["index"] for r in pools["output_nfpa_40"]["rows"]}
        if len(reference_hashes) != len(originals):
            raise ValueError("Reference image hashes are not unique")
        for condition, pool in pools.items():
            pairs = indexed(pool["rows"])
            quality = indexed(read(QUALITY / family / (condition + ".json"))["rows"])
            if set(pairs) != set(originals) or set(quality) != set(originals):
                raise ValueError("Quality and reference IDs differ")
            if any(quality[i]["label"] != originals[i]["label"] for i in originals):
                raise ValueError("Class ordering differs between paired evaluations")
            if condition.startswith("weight_"):
                score_file = Path(next(iter(pairs.values()))["candidate"]).parent / "samples.json"
                scores = indexed(read(score_file)["rows"])
            else:
                rows = read(NFPA / case_name / (condition.removeprefix("output_") + "_scores.json"))
                scores = {}
                for row in rows:
                    if row["cohort"] == "marked":
                        index = reference_hashes[input_hashes[row["id"]]]
                        if index in scores:
                            raise ValueError("Repeated NFPA reference")
                        scores[index] = row
            if set(scores) != set(originals):
                raise ValueError("Owner decisions do not align with quality measurements")
            decisions = {i: r["bit_accuracy"] > threshold for i, r in scores.items()}
            observed = sum(decisions.values())
            if observed != pool["detection"]["count"]:
                raise ValueError("Frozen owner decisions do not reproduce the published count")
            for lpips in [.1, .2, .3, .4, 1.0]:
                bounds = {"lpips": [0., lpips], "clip_image_cosine": [.9, 1.000001],
                          "ssim": [.6, 1.000001], "class_alignment_retention": [.95, None]}
                gates = {}
                for i, row in quality.items():
                    reference = row["reference_clip_alignment"]
                    retention = row["clip_alignment"] / reference if reference > 0 else None
                    gates[i] = numeric_gate({**row, "class_alignment_retention": retention}, bounds)
                results.append({"family": family, "condition": condition, "bounds": bounds,
                                **joint_outcomes(original_decisions, decisions, gates)})
            gates = {}
            component_passes = {metric: 0 for metric in current_rule}
            for i, row in quality.items():
                reference = row["reference_clip_alignment"]
                retention = row["clip_alignment"] / reference if reference > 0 else None
                values = {**row, "class_alignment_retention": retention}
                gates[i] = numeric_gate(values, current_rule)
                for metric, interval in current_rule.items():
                    component_passes[metric] += numeric_gate(values, {metric: interval})["passed"]
            current_results.append({
                "family": family, "condition": condition, "bounds": current_rule,
                "component_passes": component_passes,
                "mean_class_clip": sum(row["clip_alignment"] for row in quality.values()) / len(quality),
                **joint_outcomes(original_decisions, decisions, gates),
            })
    receipt = {"status": "Post-hoc diagnostic, not a prespecified confirmation or a validated universal quality rule.",
               "interpretation": "These conservative automatic checks measure paired similarity and class alignment, not human-verified identity, factual detail or population image quality.",
               "sources": {k: v for k, v in sources.items() if k != str(current_rule_path.relative_to(ROOT))}, "results": results}
    (ROOT / "tex/provenance/image_preservation_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    current_receipt = {
        "status": "Post-hoc application of the retained three-metric rule to the saved 512-pair cohorts.",
        "rule_file": str(current_rule_path.relative_to(ROOT)),
        "rule_sha256": hashlib.sha256(current_rule_path.read_bytes()).hexdigest(),
        "sources": sources,
        "historical_receipt": "tex/provenance/image_preservation_audit.json",
        "results": current_results,
        "scope": "Paired-content preservation and class-alignment diagnostics. These do not measure population realism or certify human quality.",
    }
    (ROOT / "tex/provenance/retained_image_preservation_audit.json").write_text(
        json.dumps(current_receipt, indent=2) + "\n")
    labels = {
        "weight_unmodified_supplied": "Original marked generator",
        "output_resize_only": "Resize only",
        **{f"weight_merge_{fraction}": f"Merge, {100*fraction:g}" + r"\% marked"
           for fraction in [.75, .5, .25, 0]},
        **{f"weight_clean_{kind}_{steps}": f"Clean {kind.title()}, {steps:,} updates"
           for kind in ["lora", "full"] for steps in [500, 2500]},
        **{f"output_nfpa_{delta}": f"NFPA, strength {delta}" for delta in [0, 20, 40]},
    }
    labels = {key: value.replace("Lora", "LoRA") for key, value in labels.items()}
    caption = (
        r"Saved image-watermark removal experiments using DistSeal RAR-XL at 256 pixels and "
        r"UViT-H/DC-AE at 512 pixels, with 512 matched inputs per row. "
        r"Access: W exposes marked target weights; clean training additionally uses clean fitting data. W+C additionally requires a compatible clean checkpoint; O uses only saved target images; -- is the baseline. "
        r"Initially detected counts the source images flagged by the unchanged owner extractor. "
        r"Removed counts their candidates that become undetected, regardless of preservation. "
        r"Pair-rule pass counts all candidates satisfying the retained rule: LPIPS $\leq0.2$, "
        r"image--image CLIP $\geq0.9$, class-alignment retention $\geq0.9$, and finite metrics. "
        r"Joint counts the intersection of removal and pair-rule pass among initially detected sources. "
        r"Every count uses the full 512-input cohort. Detection uses strict bit accuracy $>0.640625$ "
        r"for RAR and $>0.6875$ for DC-AE. The former SSIM-gated rule is archived separately. "
        r"These are post-hoc paired-content proxies; class-valid scene changes can fail them."
    )
    lines = [
        r"{\small\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{lp{.27\linewidth}lrrrr}",
        r"\caption{" + caption + r"}\label{tab:retained-image-preservation}\\",
        r"\toprule Model & Intervention & Access & \shortstack{Initially\\detected} & Removed & \shortstack{Pair-rule\\pass} & Joint\\\midrule\endfirsthead",
        r"\multicolumn{7}{l}{\tablename\ \thetable\ continued; all counts out of 512.}\\",
        r"\toprule Model & Intervention & Access & \shortstack{Initially\\detected} & Removed & \shortstack{Pair-rule\\pass} & Joint\\\midrule\endhead",
    ]
    for result in current_results:
        name = "RAR" if result["family"] == "rar" else "DC-AE"
        condition = result["condition"]
        access = "O" if condition.startswith("output_") else "W+C" if "merge" in condition else "--" if condition == "weight_unmodified_supplied" else "W"
        lines.append(
            f"{name} & {labels[result['condition']]} & {access} & {result['originally_detected']} & "
            f"{result['raw_removal']['count']} & {result['preservation']['count']} & "
            f"{result['joint_removal']['count']} " + r"\\")
    lines.append(r"\bottomrule\end{longtable}}")
    (ROOT / "tex/generated/quality_qualified_images.tex").write_text("\n".join(lines) + "\n")
    lines = [
        r"\begin{table}[ht]\centering\small",
        r"\begin{tabular}{lrrrr}\toprule",
        r"RAR intervention & Removed & \shortstack{LPIPS\\pass} & \shortstack{Image CLIP\\pass} & \shortstack{Class retention\\pass}\\\midrule",
    ]
    for result in current_results:
        if result["family"] != "rar" or result["condition"] not in {
                "weight_clean_lora_2500", "weight_clean_full_2500", "output_nfpa_0", "output_nfpa_20", "output_nfpa_40"}:
            continue
        parts = result["component_passes"]
        lines.append(
            f"{labels[result['condition']]} & {result['raw_removal']['count']} & "
            f"{parts['lpips']} & {parts['clip_image_cosine']} & {parts['class_alignment_retention']} " + r"\\")
    lines += [
        r"\bottomrule\end{tabular}",
        r"\caption{RAR-XL watermark attacks on the same 512 saved 256-pixel source images. Removed counts owner non-detections among 512 initially detected originals. Each other column applies one component of the retained proxy independently: LPIPS $\leq0.2$, image--image CLIP $\geq0.9$, or class-alignment retention $\geq0.9$. Their counts are not intersected. Class retention can pass while a different sampled scene fails paired similarity. Clean LoRA/full training requires marked target weights; NFPA has output-only target access. These metrics do not establish population realism or human-rated quality.}\label{tab:rar-proxy-components}\end{table}",
    ]
    (ROOT / "tex/generated/rar_proxy_components.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps([{key: r[key] for key in ("family", "condition", "preservation", "joint_removal")}
                      for r in current_results], indent=2))


if __name__ == "__main__":
    main()
