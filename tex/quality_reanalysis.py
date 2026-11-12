"""Recompute preservation-qualified removal from frozen images and owner scores."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from ai_content_detector.watermarking.preservation import joint_outcomes, numeric_gate


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reanalyze(path, rule):
    config = read(path)
    root = Path(config["directory"])
    result = read(root / "owner_results.json")
    freeze = read(root / "final_artifact_freeze.json")
    if result["signature"] != config["signature"] or freeze["signature"] != config["signature"]:
        raise ValueError("Owner protocol signature changed")
    if not config["protocol"].get("quality_fallback"):
        raise ValueError("This reanalysis requires the earlier quality-fallback study")
    summaries, vectors = {}, {}
    for name in config["cases"]:
        selection = root / name / "selection.json"
        if sha(selection) != result["selection_hashes"][name] or sha(selection) != freeze["selection_hashes"][name]:
            raise ValueError("Frozen selection changed")
        quality_path = root / name / "final-attack/quality.json"
        if sha(quality_path) != freeze["raw_quality_hashes"][name]:
            raise ValueError("Raw quality measurements changed")
        rows = read(quality_path)
        ids = [str(r["index"]) for r in rows]
        if ids != [str(i) for i in range(config["protocol"]["final"]["n"])]:
            raise ValueError("Quality indices changed")
        for kind in ["base", "raw_attack"]:
            images = freeze["samples"][name + ":" + kind]
            if [str(r["index"]) for r in images] != ids:
                raise ValueError("Image indices changed")
            if [r["label"] for r in images] != config["protocol"]["final"]["labels"]:
                raise ValueError("Image class alignment changed")
            for image in images:
                if sha(image["path"]) != image["sha256"]:
                    raise ValueError("Frozen source or raw PNG changed")
        threshold = result["owner_threshold"]
        base = dict(zip(ids, [bool(x > threshold) for x in result["bit_accuracies"][name + ":base"]]))
        raw = dict(zip(ids, [bool(x > threshold) for x in result["bit_accuracies"][name + ":raw_attack"]]))
        old_gates = {str(r["index"]): numeric_gate(r, config["protocol"]["quality_rule"]) for r in rows}
        if any(old_gates[str(r["index"])] != r["preservation"] for r in rows):
            raise ValueError("Old quality gate cannot be reconstructed")
        old = joint_outcomes(base, raw, old_gates)
        if old != result["raw_conditions"][name]:
            raise ValueError("Old raw owner outcome cannot be reconstructed")
        old_returned = {i: raw[i] if old_gates[i]["passed"] else base[i] for i in ids}
        returned_gates = {i: {"passed": True, "failures": []} for i in ids}
        if joint_outcomes(base, old_returned, returned_gates) != result["conditions"][name]:
            raise ValueError("Old fallback outcome cannot be reconstructed")
        gates = {str(r["index"]): numeric_gate(r, rule["quality_rule"]) for r in rows}
        revised = joint_outcomes(base, raw, gates)
        returned = {i: raw[i] if gates[i]["passed"] else base[i] for i in ids}
        gated = joint_outcomes(base, returned, returned_gates)
        if gated["joint_removal"] != revised["joint_removal"]:
            raise ValueError("Gated and raw joint counts disagree")
        vectors[name] = np.array([base[i] and not raw[i] and gates[i]["passed"] for i in ids], dtype=float)
        summaries[name] = {"old_joint_removal": old["joint_removal"], "old_raw_preservation": old["preservation"],
                           "revised_raw": revised, "revised_returned": gated,
                           "newly_passing_ids": [i for i in ids if gates[i]["passed"] and not old_gates[i]["passed"]],
                           "newly_rejected_ids": [i for i in ids if old_gates[i]["passed"] and not gates[i]["passed"]],
                           "raw_gates": gates}
    rng = np.random.default_rng(2026110919)
    n = len(vectors["original"])
    draws = rng.integers(n, size=(10000, n))
    differences = {}
    for name in ["fixed", "adapted"]:
        delta = vectors[name] - vectors["original"]
        differences[name] = {"removal_difference": float(delta.mean()),
                             "paired_bootstrap_95_percent_interval": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
                             "scope": "Descriptive paired resampling, not a prespecified confirmatory test."}
    return {"manifest": str(Path(path).resolve()), "manifest_sha256": sha(path),
            "owner_results_sha256": sha(root / "owner_results.json"),
            "final_freeze_sha256": sha(root / "final_artifact_freeze.json"),
            "original_gate_reproduction": "Exact raw and returned outcome reconstruction passed for every case.",
            "conditions": summaries, "paired_differences": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rule")
    parser.add_argument("horizon100")
    parser.add_argument("horizon500")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    if args.suffix and not re.fullmatch(r"[a-z0-9_]+", args.suffix):
        parser.error("Suffix must contain lowercase letters, digits or underscores")
    stem = "revised_quality" + ("_" + args.suffix if args.suffix else "") + "_results"
    rule = read(args.rule)
    for name in ["calibration", "grid"]:
        if sha(ROOT / rule[name]) != rule[name + "_sha256"]:
            raise ValueError("Calibration inputs changed after freezing the rule")
    results = {"rule": str(Path(args.rule).resolve()), "rule_sha256": sha(args.rule),
               "scope": rule["scope"], "studies": {name: reanalyze(path, rule) for name, path in
                                                   [("horizon100", args.horizon100), ("horizon500", args.horizon500)]}}
    output = ROOT / "tex/provenance" / (stem + ".json")
    output.write_text(json.dumps(results, indent=2) + "\n")
    rows = [r"\begin{table}[ht]\centering\small", r"\begin{tabular}{llrrr}\toprule",
            r"\shortstack{Inner\\steps} & Model & \shortstack{Historical\\joint} & \shortstack{Retained\\joint} & \shortstack{Candidate\\pass}\\\midrule"]
    for horizon, study in results["studies"].items():
        for name, value in study["conditions"].items():
            new = value["revised_raw"]
            n = new["n"]
            rows.append(f"{horizon.removeprefix('horizon')} & {name} & "
                        f"{value['old_joint_removal']['count']}/{n} & {new['joint_removal']['count']}/{n} & "
                        f"{new['preservation']['count']}/{n}" + r"\\")
        print(horizon, {name: value["revised_raw"]["joint_removal"]["count"]
                        for name, value in study["conditions"].items()}, flush=True)
    rows += [r"\bottomrule\end{tabular}", r"\caption{" + 'Post-hoc preservation reanalysis of open-weight DistSeal UViT-H/DC-AE clean-training attacks, with 64 paired inputs per row. Inner steps gives the clean adaptation horizon used in training the defense candidates. Original is the released marked generator; Fixed and Adapted are the meta-distilled reference variants. Historical joint uses LPIPS $\\leq0.2$, SSIM $\\geq0.6$, image CLIP $\\geq0.9$ and class retention $\\geq0.95$. Retained joint uses LPIPS $\\leq0.2$, image CLIP $\\geq0.9$ and class retention $\\geq0.9$. Candidate pass counts raw adapted images passing the retained rule before any source fallback. Joint counts simultaneous preservation and loss of owner detection at bit accuracy $\\leq0.6875$. The two horizons have different generation seeds; comparisons are paired within each horizon. This reuses evaluated data.' + r"}\end{table}"]
    (ROOT / "tex/generated" / (stem + ".tex")).write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
