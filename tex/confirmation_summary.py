"""Combine frozen, disjoint text confirmation shards with paired uncertainty."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

from development_summary import ROOT, read, save, sha, table, text_summary

sys.path.insert(0, str(ROOT / "scripts"))
from publish_watermark_results import binomial_ci
from ai_content_detector.watermarking.preservation import joint_outcomes


def summarize(manifests):
    originals, attacked, gates, shards, null_ids = {}, {}, {}, [], set()
    calibration_ids, shard_ids, protocol = set(), set(), None
    owner_source = ROOT / "tex/provenance/code/score_textseal_exposure_confirmation.py"
    for manifest in manifests:
        config = read(manifest)
        confirmation = config["confirmation"]
        shard = confirmation["shard"]
        if shard in shard_ids:
            raise ValueError("Repeated confirmation shard")
        shard_ids.add(shard)
        current = {k: config[k] for k in ["watermark", "sampling", "quality_rule", "selection", "attack_arms"]}
        if protocol is not None and current != protocol:
            raise ValueError("Confirmation protocols differ")
        protocol = current
        text_summary(manifest)
        root, generated = Path(config["directory"]), Path(config["generated"])
        result, selected = read(root/"owner_results.json"), read(root/"selection.json")["selected"]
        if result["scorer_source_sha256"] != sha(owner_source):
            raise ValueError("Archived owner source differs from the executed source")
        rows = read(generated/"originals.json")
        ids = [r["id"] for r in rows]
        expected = [r["id"] for r in config["data"]["evaluation"]]
        if ids != expected or len(set(ids)) != len(ids) or set(ids) & set(originals):
            raise ValueError("Confirmation sources overlap or are misaligned")
        calibration_ids.update(r["id"] for r in config["data"]["calibration"])
        null_ids.update(sha_text(r["human_text"]) for r in config["data"]["evaluation"])
        originals.update({key: score["p_value"] <= .001 for key, score in result["original_scores"].items()})
        for arm, choices in selected.items():
            scores = {(r["id"], r["attempt"]): r for r in result["attempt_scores"][arm]}
            quality = {(r["id"], r["attempt"]): r for r in read(root/(arm+"-quality.json"))}
            decisions, preservation = attacked.setdefault(arm, {}), gates.setdefault(arm, {})
            for choice in choices:
                pair = choice["id"], choice["attempt"]
                decisions[pair[0]] = scores[pair]["p_value"] <= .001
                preservation[pair[0]] = {k: quality[pair][k] for k in ["passed", "failures"]}
        shards.append({"manifest": str(Path(manifest).resolve()), "owner_results_sha256": sha(root/"owner_results.json"),
                       "n": len(ids), "shard": shard, "null_detected": result["human_null"]["detected"]})
    if shard_ids != {0, 1} or len(originals) != 128 or set(originals) & calibration_ids:
        raise ValueError("Expected disjoint 128-source confirmation and separate calibration")
    conditions = {arm: joint_outcomes(originals, attacked[arm], gates[arm]) for arm in attacked}
    for values in conditions.values():
        for field in ["preservation", "raw_removal", "joint_removal", "removal_given_preservation"]:
            value = values[field]
            value["ci95"] = binomial_ci(value["count"], value["n"]) if value["n"] else None
    ids = sorted(originals)
    rng = np.random.default_rng(2026110802)
    indices = rng.integers(len(ids), size=(10000, len(ids)))
    paired = {}
    for attack in ["direct", "smooth", "ratio"]:
        before, after = "pre_"+attack, "conditional_"+attack
        a = np.asarray([i in conditions[before]["joint_success_ids"] for i in ids])
        b = np.asarray([i in conditions[after]["joint_success_ids"] for i in ids])
        differences = b.astype(float)-a
        lost, gained = int((a & ~b).sum()), int((~a & b).sum())
        paired[attack] = {"conditional_minus_pre": float(differences.mean()),
                          "paired_percentile_bootstrap_ci95": np.quantile(differences[indices].mean(1), [.025,.975]).tolist(),
                          "pre_only_joint_success": lost, "conditional_only_joint_success": gained,
                          "exact_mcnemar_p": float(binomtest(gained, gained+lost, .5).pvalue) if gained+lost else 1.}
    receipt = {"shards": shards, "n": len(ids), "original_detected": sum(originals.values()),
               "calibration_id_count": len(calibration_ids), "human_null_unique_texts": len(null_ids),
               "human_null_detected": sum(s["null_detected"] for s in shards),
               "conditions": conditions, "paired_joint_differences": paired,
               "analysis_source_sha256": sha(__file__), "bootstrap_seed": 2026110802, "bootstrap_replicates": 10000,
               "uncertainty_scope": "Post-hoc paired prompt bootstrap and unadjusted descriptive exact McNemar tests; fixed single key, conditional on these models and source population. Not training-seed or key uncertainty."}
    rows = [["Interface / client", "Preserved", "Removed", "Joint removed", r"Joint 95\% CI"]]
    for arm, values in conditions.items():
        fraction = lambda field: str(values[field]["count"])+"/"+str(values[field]["n"])
        low, high = values["joint_removal"]["ci95"]
        rows.append([arm.replace("conditional_", "Conditional / ").replace("pre_", "Pre / "),
                     fraction("preservation"), fraction("raw_removal"), fraction("joint_removal"),
                     f"[{100*low:.1f}, {100*high:.1f}]"+r"\%"])
    caption = 'Fresh-source simulated probability-only confirmation for native TextSeal 2 on Qwen3.5-4B instruct, 128 HC3 questions and one frozen watermark key. All 128 source answers are detected at $p\\leq0.001$. Pre exposes underlying pre-watermark probabilities; Conditional exposes their estimated keyed sampler law. Direct resamples, smooth mixes an independent Qwen3.5-0.8B reference, and ratio reweights advertised tokens using that reference. Preserved counts outputs passing the frozen text rule; Removed counts owner non-detections regardless of preservation; Joint counts both outcomes. Every removal fraction uses the 128 originally detected sources. Selection takes the first quality-passing attempt among three, before owner scoring. Intervals are exact binomial 95\\% intervals conditional on these models and key.'
    (ROOT/"tex/generated/exposure_confirmation_results.tex").write_text(table(rows,"lrrrr",caption))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "serif", "font.size": 8}):
        fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.5), sharey=True)
        for ax, field, title in zip(axes, ["preservation", "raw_removal", "joint_removal"],
                                   ["Preservation", "Raw removal", "Joint removal"]):
            for offset, prefix, color, label in [(-.18,"pre_","#3174a1","Pre-watermark"),
                                                 (.18,"conditional_","#d4782f","Conditional")]:
                values = [conditions[prefix+attack][field] for attack in ["direct", "smooth", "ratio"]]
                rates = np.asarray([v["rate"] for v in values])
                intervals = np.asarray([v["ci95"] for v in values])
                ax.bar(np.arange(3)+offset, 100*rates, .34, color=color, label=label,
                       yerr=100*np.stack((rates-intervals[:,0], intervals[:,1]-rates)),
                       error_kw={"elinewidth":.7,"capsize":2})
            ax.set_title(title)
            ax.set_xticks(range(3), ["Direct", "Smoothing", "Ratio"])
            ax.set_ylim(0, 108)
            ax.set_yticks([0,25,50,75,100])
            ax.spines[["top","right"]].set_visible(False)
        axes[0].set_ylabel("Percent of 128 sources")
        axes[1].legend(loc="lower center", bbox_to_anchor=(.5,1.16), ncol=2, frameon=False)
        fig.tight_layout()
        for extension in ["pdf", "png"]:
            filename = ROOT/f"tex/figures/exposure-confirmation.{extension}"
            fig.savefig(filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
    save(ROOT/"tex/provenance/text_exposure_confirmation_results.json", receipt)
    print(json.dumps({"original_detected": receipt["original_detected"], "paired": paired}, indent=2))


def sha_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs=2)
    summarize(parser.parse_args().manifests)
