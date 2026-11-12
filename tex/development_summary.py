"""Publish frozen development owner results with aligned preservation evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import unicodedata

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)+"\n")


def escaped(text):
    replacements = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$",
                    "&": r"\&", "#": r"\#", "_": r"\_", "%": r"\%", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}", chr(8212): "[U+2014]"}
    return "".join(replacements.get(c, c if ord(c) < 0x2000 or c in "‘’“”–" else f"[U+{ord(c):04X}]") for c in text)


def table(rows, columns, caption):
    return "\n".join([r"\begin{table}[ht]\centering\small", r"\begin{tabular}{"+columns+r"}\toprule",
                       *[" & ".join(row)+r"\\"+(r"\midrule" if i == 0 else "") for i, row in enumerate(rows)],
                       r"\bottomrule\end{tabular}", r"\caption{"+caption+r"}\end{table}", ""])


def text_summary(manifest):
    config = read(manifest)
    directory = Path(config["directory"])
    generated = Path(config["generated"])
    result = read(directory/"owner_results.json")
    complete = read(directory/"quality_complete.json")
    if sha(manifest) != result["quality_manifest_sha256"] or sha(directory/"selection.json") != result["selection_sha256"]:
        raise ValueError("Frozen text owner inputs changed")
    originals = read(generated/"originals.json")
    original_by_id = {r["id"]: r for r in originals}
    selected = read(directory/"selection.json")["selected"]
    names = {"pre_direct": "Pre / direct", "pre_smooth": "Pre / smoothing",
             "conditional_direct": "Conditional / direct", "conditional_smooth": "Conditional / smoothing",
             "pre_ratio": "Pre / ratio tilt", "conditional_ratio": "Conditional / ratio tilt"}
    for guard in ["guard10", "guard20"]:
        for attack in ["direct", "smooth", "ratio"]:
            names[guard+"_"+attack] = guard.replace("guard", "Prune ")+" / "+attack
    names["reference_only"] = "Independent reference"
    rows = [["Exposed law / attack", "Access", "Preserved", "Removed", "Joint removed", "Word copies"]]
    receipt = {"manifest": str(Path(manifest).resolve()), "owner_results_sha256": sha(directory/"owner_results.json"),
               "original_detected": result["original_detected"], "conditions": {},
               "overlap_scope": "Saved generation token IDs, not retokenized text; descriptive, not a calibrated hypothesis test."}
    chosen = {}
    for arm, choices in selected.items():
        quality_path = directory/(arm+"-quality.json")
        if sha(quality_path) != complete["quality_hashes"][arm]:
            raise ValueError("Text quality changed")
        attempts = {(r["id"], r["attempt"]): r for r in read(generated/(arm+".json"))}
        quality = {(r["id"], r["attempt"]): r for r in read(quality_path)}
        owner_scores = {(r["id"], r["attempt"]): r for r in result["attempt_scores"][arm]}
        copies, overlaps, fallbacks = 0, [], 0
        chosen[arm] = {}
        for selection in choices:
            key = selection["id"], selection["attempt"]
            row, gate = attempts[key], quality[key]
            if hashlib.sha256(row["text"].encode()).hexdigest() != selection["candidate_sha256"] or selection["passed"] != gate["passed"]:
                raise ValueError("Text selection alignment changed")
            original = original_by_id[row["id"]]
            words = lambda text: re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
            copies += words(row["text"]) == words(original["text"])
            pairs = lambda ids: {tuple(ids[i:i+4]) for i in range(len(ids)-3)}
            current = pairs(row["tokens"])
            overlaps.append(len(current & pairs(original["tokens"]))/len(current) if current else 0.)
            fallbacks += row.get("source_blocking_fallbacks", 0)
            chosen[arm][row["id"]] = {"text": row["text"], "passed": gate["passed"], "failures": gate["failures"],
                                      "owner_p_value": owner_scores[key]["p_value"],
                                      "attempt": row["attempt"], "source_blocking_fallbacks": row.get("source_blocking_fallbacks", 0)}
        scores = result["conditions"][arm]
        if sum(r["passed"] for r in chosen[arm].values()) != scores["preservation"]["count"]:
            raise ValueError("Text preservation aggregate changed")
        fraction = lambda field: f"{scores[field]['count']}/{scores[field]['n']}"
        rows.append([names[arm], "O" if arm == "reference_only" else "P", fraction("preservation"), fraction("raw_removal"), fraction("joint_removal"), f"{copies}/{len(choices)}"])
        receipt["conditions"][arm] = {**scores, "normalized_word_copies": copies,
                                     "mean_saved_four_pair_overlap": statistics.mean(overlaps),
                                     "selected_source_blocking_fallbacks": fallbacks,
                                     "all_attempt_source_blocking_fallbacks": sum(r.get("source_blocking_fallbacks", 0) for r in attempts.values())}
    shard = config.get("confirmation", {}).get("shard")
    suffix = ("guard" if config.get("exposure_guard") else "confirmation_"+str(shard)
              if shard is not None else "ratio" if "pre_ratio" in selected else "blocked")
    caption = 'TextSeal 2 simulated probability-only development experiment with Qwen3.5-4B instruct on 32 HC3 source answers, all originally detected at native $p\\leq0.001$. Pre uses underlying probabilities; Conditional uses the estimated keyed sampler law. Access: P is simulated probability-only target access; O uses only the saved marked answer. The independent reference is Qwen3.5-0.8B. Preserved counts first-quality-passing selections under Jina cosine $\\geq0.9$, bidirectional untruncated entailment, exact number sets, finite perplexity no greater than 1.5 times the original, and completion/repetition checks. Removed counts selected outputs evading owner detection regardless of quality; Joint counts preservation and removal together. Word copies counts normalized source equality. All fractions retain the 32-input denominator. Three attempts are frozen before quality selection and owner scoring. These sources are reused development data.'
    caption = caption.replace('32 HC3', f'{len(originals)} HC3').replace('32-input', f'{len(originals)}-input')
    if shard is not None:
        caption = caption.replace('development experiment', f'fresh-source confirmation shard {shard}').replace('These sources are reused development data.', 'The two confirmation shards contain disjoint question identities.')
    if config.get("exposure_guard"):
        caption += ' Prune 10 and Prune 20 retain at least 90\\% and 80\\% of the ordinary top-ten estimated probability mass. Independent reference rewrites outside the advertised support.'
    (ROOT/f"tex/generated/exposure_{suffix}_results.tex").write_text(table(rows, "llrrrr",
        caption))
    positions = [0, 8, 16, 24]
    qualitative = []
    display_arms = list(selected) if suffix in ["ratio", "guard"] else ["pre_smooth", "conditional_smooth"]
    for position in positions:
        row = originals[position]
        qualitative += [r"\subsection*{Source position "+str(position)+"}", escaped(row["question"]), "",
                        r"\paragraph{Original; $p="+f"{result['original_scores'][row['id']]['p_value']:.2e}"+r"$.} "+escaped(row["text"]), ""]
        for arm in display_arms:
            choice = chosen[arm][row["id"]]
            status = "passes" if choice["passed"] else "fails"
            qualitative += [r"\paragraph{"+names[arm]+f"; quality {status}; $p={choice['owner_p_value']:.2e}$."+"} "+escaped(choice["text"]), ""]
    (ROOT/f"tex/generated/exposure_{suffix}_qualitative.tex").write_text("\n".join(qualitative))
    receipt["qualitative_positions"] = positions
    receipt["position_selection"] = "Every eighth source position after the run, without filtering quality or detection; not preregistered. Full displayed texts are preserved, with explicit Unicode fallback where needed."
    save(ROOT/f"tex/provenance/text_exposure_{suffix}_results.json", receipt)
    save(ROOT/f"tex/provenance/text_exposure_{suffix}_samples.json", {"originals": originals, "selected": chosen})
    print(json.dumps({k: {f: v[f] for f in ["preservation", "joint_removal", "normalized_word_copies"]} for k, v in receipt["conditions"].items()}))


def image_summary(manifest, suffix=None):
    config = read(manifest)
    stem = "preserving_fallback" if config["protocol"].get("quality_fallback") else "preserving_attack"
    if suffix is not None:
        if not re.fullmatch(r"[a-z0-9_]+", suffix):
            raise ValueError("Image publication suffix must contain lowercase letters, digits or underscores")
        stem += "_"+suffix
    root = Path(config["directory"])
    result = read(root/"owner_results.json")
    freeze = read(root/"final_artifact_freeze.json")
    if result["signature"] != config["signature"] or result["selection_hashes"] != freeze["selection_hashes"]:
        raise ValueError("Frozen image owner inputs changed")
    rows = [["Starting model", "Updates", "Anchor", "Preserved", "Removed", "Joint removed"]]
    for name, scores in result["conditions"].items():
        selection_path = root/name/"selection.json"
        if sha(selection_path) != result["selection_hashes"][name]:
            raise ValueError("Image selection changed")
        selected = read(selection_path)["selected"]
        attack_folder = "final-gated" if config["protocol"].get("quality_fallback") else "final-attack"
        quality_path = root/name/attack_folder/"quality.json"
        if sha(quality_path) != freeze["quality_hashes"][name]:
            raise ValueError("Final image quality changed")
        quality = read(quality_path)
        samples = {kind: read(root/name/(attack_folder if kind == "attack" else "final-base")/"samples.json")["rows"] for kind in ["base", "attack"]}
        for kind, pool in samples.items():
            if pool != freeze["samples"][name+":"+kind] or [r["index"] for r in pool] != list(range(64)):
                raise ValueError("Final image sample alignment changed")
            for row in pool:
                if sha(row["path"]) != row["sha256"]:
                    raise ValueError("Final image PNG changed")
        fraction = lambda field: f"{scores[field]['count']}/{scores[field]['n']}"
        rows.append([name.capitalize(), str(selected.get("step", 0)), str(selected.get("anchor_weight", "--")),
                     fraction("preservation"), fraction("raw_removal"), fraction("joint_removal")])
        tile, heading = 256, 56
        grid = Image.new("RGB", (4*tile, 4*(tile+heading)), "white")
        draw = ImageDraw.Draw(grid)
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
        for number, index in enumerate(range(0, 64, 8)):
            y, pair = divmod(number, 2)
            for side, kind in enumerate(["base", "attack"]):
                sample = samples[kind][index]
                if sample["label"] != samples["base"][index]["label"]:
                    raise ValueError("Qualitative image labels changed")
                x = 2*pair+side
                with Image.open(sample["path"]) as image:
                    grid.paste(image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS), (x*tile, y*(tile+heading)+heading))
                bit = result["bit_accuracies"][name+":"+kind][index]
                label = f"{kind}; index {index}; class {sample['label']}\nbit {bit:.3f}"
                if kind == "attack":label += f"; LPIPS {quality[index]['lpips']:.3f}\nquality {'pass' if quality[index]['preservation']['passed'] else 'FAIL'}"
                draw.text((x*tile+3, y*(tile+heading)+2), label, font=font, fill="black")
        grid.save(ROOT/f"tex/figures/{stem.replace('_', '-')}-{name}.png")
        if config["protocol"].get("quality_fallback"):
            raw_quality_path = root/name/"final-attack/quality.json"
            if sha(raw_quality_path) != freeze["raw_quality_hashes"][name]:
                raise ValueError("Raw image quality changed")
            raw_quality = read(raw_quality_path)
            raw = read(root/name/"final-attack/samples.json")["rows"]
            if raw != freeze["samples"][name+":raw_attack"]:
                raise ValueError("Raw image sample alignment changed")
            for row in raw:
                if sha(row["path"]) != row["sha256"]:
                    raise ValueError("Raw image PNG changed")
            triplets = Image.new("RGB", (3*tile, 4*(tile+heading)), "white")
            draw = ImageDraw.Draw(triplets)
            for number, index in enumerate(range(0,64,16)):
                for side, (kind,pool) in enumerate([("base",samples["base"]),("raw_attack",raw),("attack",samples["attack"])]):
                    sample = pool[index]
                    if sample["index"] != index or sample["label"] != samples["base"][index]["label"]:
                        raise ValueError("Raw triplet sample IDs or labels changed")
                    with Image.open(sample["path"]) as source:
                        triplets.paste(source.convert("RGB").resize((tile,tile),Image.Resampling.LANCZOS),
                                       (side*tile,number*(tile+heading)+heading))
                    bit = result["bit_accuracies"][name+":"+kind][index]
                    label = {"base":"Source","raw_attack":"Raw adaptation","attack":"Returned output"}[kind]
                    label += f"; index {index}\nbit {bit:.3f}"
                    if kind=="raw_attack":
                        label += f"; LPIPS {raw_quality[index]['lpips']:.3f}\nquality {'pass' if raw_quality[index]['preservation']['passed'] else 'FAIL'}"
                    draw.text((side*tile+3,number*(tile+heading)+2),label,font=font,fill="black")
            triplets.save(ROOT/f"tex/figures/{stem.replace('_','-')}-triplets-{name}.png")
    caption = 'Independent open-weight clean-training attacks against DistSeal UViT-H/DC-AE at 512 pixels. Original is the released marked generator; Fixed and Adapted are the two prediction-anchored meta-distilled variants. Updates counts clean attacker updates, and Anchor is its prediction-retention coefficient. Preserved counts returned outputs passing LPIPS $\\leq0.2$, SSIM $\\geq0.6$, image CLIP $\\geq0.9$ and class-alignment retention $\\geq0.95$. Removed counts returned images losing owner detection, before requiring preservation. Joint counts both outcomes. All fractions use 64 reserved final inputs per starting model, initially all detected at strict bit accuracy $>0.6875$. Checkpoint selection uses 32 separate validation inputs without owner feedback.'
    if config["protocol"].get("quality_fallback"):
        caption = 'Independent open-weight clean-training attacks against DistSeal UViT-H/DC-AE at 512 pixels. Original is the released marked generator; Fixed and Adapted are the two prediction-anchored meta-distilled variants. Updates counts clean attacker updates, and Anchor is its prediction-retention coefficient. Preserved counts returned outputs passing LPIPS $\\leq0.2$, SSIM $\\geq0.6$, image CLIP $\\geq0.9$ and class-alignment retention $\\geq0.95$. Removed counts returned images losing owner detection, before requiring preservation. Joint counts both outcomes. All fractions use 64 reserved final inputs per starting model, initially all detected at strict bit accuracy $>0.6875$. Checkpoint selection uses 32 separate validation inputs without owner feedback. This experiment returns the marked source when the raw adapted image fails preservation. The reported counts concern returned outputs; the raw adapted generator can fail the pair rule. Updates and anchor coefficients are selected separately for each starting model.'
    (ROOT/f"tex/generated/{stem}_results.tex").write_text(table(rows, "lrrrrr", caption))
    save(ROOT/f"tex/provenance/{stem}_results.json", {"manifest": str(Path(manifest).resolve()),
         "owner_results_sha256": sha(root/"owner_results.json"), "results": result,
         "validation": "Verified base/returned PNG hashes, sample indices, labels, selection and final quality hashes; fallback studies also verify all raw PNGs and raw quality hashes.",
         "qualitative_positions": list(range(0, 64, 8)),
         "raw_triplet_positions": list(range(0,64,16)) if config["protocol"].get("quality_fallback") else [],
         "position_selection": "Every eighth final position after the run; no outcome filtering and no preregistration claim."})
    print(json.dumps(result["conditions"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modality", choices=["text", "image"])
    parser.add_argument("manifest")
    parser.add_argument("--suffix")
    args = parser.parse_args()
    if args.modality=="image":
        image_summary(args.manifest,args.suffix)
    elif args.suffix is not None:
        parser.error("--suffix currently applies to image summaries only")
    else:
        text_summary(args.manifest)
