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
    receipt = {"status": "Post-hoc diagnostic, not a prespecified confirmation or a validated universal quality rule.",
               "interpretation": "These conservative automatic checks measure paired similarity and class alignment, not human-verified identity, factual detail or population image quality.",
               "sources": sources, "results": results}
    (ROOT / "tex/provenance/image_preservation_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    lines = [r"{\small\begin{longtable}{lp{.36\linewidth}rrrr}\toprule",
             r"Model & Condition & Initial + & Raw removed & Proxy pass & Joint\\\midrule\endhead"]
    for result in results:
        if result["bounds"]["lpips"][1] == .2:
            name = result["condition"].replace("_", r"\_\allowbreak{}")
            lines.append(f"{result['family'].upper()} & \\scriptsize {name} & {result['originally_detected']} & "
                         f"{result['raw_removal']['count']} & {result['preservation']['count']} & "
                         f"{result['joint_removal']['count']} \\\\")
    lines += [r"\bottomrule\end{longtable}}",
              r"\noindent Post-hoc image proxy: LPIPS $\leq .2$, SSIM $\geq .6$, image-image CLIP $\geq .9$ and class-text CLIP retention $\geq .95$. All 512 original outputs per model remain in the denominator; joint success also requires initial detection and final non-detection. Failure counts and LPIPS-floor sensitivity appear in the machine-readable receipt."]
    (ROOT / "tex/generated/quality_qualified_images.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps([{key: r[key] for key in ("family", "condition", "preservation", "joint_removal")}
                      for r in results if r["bounds"]["lpips"][1] == .2], indent=2))


if __name__ == "__main__":
    main()
