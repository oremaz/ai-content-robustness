"""Export fixed-position text and image comparisons from saved experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

from migrate import escaped
from quality_audit import preservation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from watermark_report_assets import font, sha, wrapped

SOURCES = {}
EXAMPLES = {}
MISSING = {int(value, 16) for value in json.loads(
    (ROOT / "tex/provenance/missing_glyphs.json").read_text())}


def read_json(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def read_saved(path):
    import torch
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def text_tex(text):
    display = "".join(
        f"[U+{ord(char):04X}]" if ord(char) in MISSING or ord(char) > 65535 or ord(char) == 0x2014 else char
        for char in text)
    return escaped(display)


def write_text_figure(name, prompt, panels, caption, label):
    width = ".48" if len(panels) == 2 else ".315"
    lines = [
        r"\begin{figure}[p]\centering",
        r"\begin{minipage}{\linewidth}\footnotesize\textbf{Prompt.} " + text_tex(prompt),
        r"\end{minipage}\par\medskip",
    ]
    for number, (title, details, text) in enumerate(panels):
        if number:
            lines.append(r"\hfill")
        lines += [
            r"\begin{minipage}[t]{" + width + r"\linewidth}\raggedright\footnotesize",
            r"\textbf{" + text_tex(title) + r"}\par\smallskip",
            text_tex(details) + r"\par\medskip",
            text_tex(text),
            r"\end{minipage}",
        ]
    lines += [r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{figure}"]
    (ROOT / "tex/generated" / (name + ".tex")).write_text("\n".join(lines) + "\n")


def detector_text():
    root = ROOT / "results/research_workflow/defender_comparison"
    base = read_json(root / "base_pairs.json")["evaluation"]
    prompts = [row["prompt"] for row in base]
    base_cache = None
    for path in sorted((root / "base_generation/generation").glob("*.pt")):
        cached = read_saved(path)
        if cached["config"]["prompts"] == prompts:
            base_cache = cached
            break
    if base_cache is None or base_cache["texts"] != [row["ai"] for row in base]:
        raise ValueError("Base generator cache does not match the saved evaluation pairs")
    branches = [
        ("decoder", 1213, "Gemma"),
        ("encoder", 1487, "ModernBERT-large"),
        ("ensemble", 502, "seven-detector ensemble"),
    ]
    paths = []
    for group, step, target in branches:
        folder = root / f"checkpoint_evaluations/{group}/step-{step}"
        candidates = list((folder / "generation").glob("*.pt"))
        if len(candidates) != 1:
            raise ValueError("Expected one retained generation cache")
        cached = read_saved(candidates[0])
        if cached["config"]["prompts"] != prompts or len(cached["texts"]) != len(base):
            raise ValueError("Attacked outputs differ from the held-out prompt order")
        config = cached["config"]
        index = 0
        name = "detector_text_" + group
        details = lambda value: (
            f"Seed {value['seed']}; maximum {value['max_new_tokens']} new tokens; "
            f"temperature {value['temperature']}; top-p {value['top_p']}.")
        caption = (
            "AI-detector evasion with Qwen3.5-9B-Base on a held-out CNN/DailyMail prefix. "
            "Left is the unchanged base generator; right is its GRPO-trained LoRA attacker "
            f"targeting {target} at update {step:,}. Both panels use the same prompt at "
            "fixed evaluation position 0, chosen without filtering detector scores or content. "
            "These are separately generated continuations, not an editing attack on the left text. "
            "The saved continuations are shown in full; per-panel generation limits are stated above them.")
        write_text_figure(
            name, base[index]["prompt"],
            [("Initial Qwen3.5-9B-Base", details(base_cache["config"]), base[index]["ai"]),
             (f"Attacked Qwen, target {target}", details(config), cached["texts"][index])],
            caption, "fig:" + name.replace("_", "-"))
        EXAMPLES[name] = {
            "position": index, "sample_id": base[index]["sample_id"],
            "target_detector": target, "attacker_update": step,
            "prompt": base[index]["prompt"], "initial": base[index]["ai"],
            "attacked": cached["texts"][index], "generation": {
                key: config[key] for key in ["seed", "max_new_tokens", "temperature", "top_p"]},
        }
        paths.append(r"\input{generated/" + name + "}")
    (ROOT / "tex/generated/detector_text_examples.tex").write_text("\n".join(paths) + "\n")


def watermark_text():
    data = read_json(ROOT / "docs/watermarking/data/examples.json")
    inputs = []
    for scheme, title in [("textseal2", "TextSeal 2"), ("latent", "local distilled text watermark")]:
        ordinary = data[f"text-{scheme}-vanilla"][0]
        bira = data[f"text-{scheme}-bira"][0]
        if ordinary["id"] != bira["id"] or ordinary["before"] != bira["before"]:
            raise ValueError("Watermark comparison does not share the same marked source")
        def description(score, quality=None):
            if scheme == "textseal2":
                detected = score["eligible"] and score["p_value"] <= .001
                value = f"Owner p-value {score['p_value']:.3g}"
            else:
                detected = score["eligible"] and score["score"] > -.6263284683
                value = f"Owner score {score['score']:.4f}"
            value += "; " + ("detected" if detected else "undetected")
            if quality is not None:
                passed = preservation(quality["quality"]) and quality["jina_cosine"] >= .9
                value += f"; paired text proxy {'pass' if passed else 'fail'}"
            return value + "."
        panels = [
            ("Initial marked Qwen3.5-4B", description(ordinary["before_score"]), ordinary["before"]),
            ("Ordinary Llama rewrite", description(ordinary["after_score"], ordinary), ordinary["after"]),
            ("BIRA Llama rewrite", description(bira["after_score"], bira), bira["after"]),
        ]
        rule = (r"TextSeal 2 flags native $p\leq0.001$." if scheme == "textseal2" else
                r"The local student's extractor flags scores $>-0.6263284683$.")
        caption = (
            f"Text watermark removal for {title} on Qwen3.5-4B instruct, thinking disabled. "
            "Columns show the initial marked answer, an ordinary rewrite and a BIRA rewrite; "
            "both attacks use independent Llama-3.1-8B-Instruct with text-only access. "
            "This is fixed position 0 of the 256-question HC3 cohort, selected without quality or detection filtering. "
            "The original generation budget is 200 tokens, so the source can be incomplete. "
            + rule + " The paired text proxy requires Jina cosine at least 0.90, bidirectional entailment "
            "without NLI truncation, exact numeric-set agreement, valid positive reference perplexity, "
            "and no wrapper leakage or long letter runs. Full saved texts are shown.")
        name = "watermark_text_" + scheme
        write_text_figure(name, ordinary["question"], panels, caption, "fig:" + name.replace("_", "-"))
        EXAMPLES[name] = {"ordinary": ordinary, "bira": bira}
        inputs.append(r"\input{generated/" + name + "}")
    (ROOT / "tex/generated/watermark_text_examples.tex").write_text("\n".join(inputs) + "\n")


def image_grid(filename, headings, rows):
    tile, gap, top = 320, 12, 68
    width = len(headings) * (tile + gap) + gap
    row_heights = []
    for row in rows:
        row_heights.append(44 + 20 * len(wrapped(
            ImageDraw.Draw(Image.new("RGB", (1, 1))), row["prompt"], font(17), width - 24)))
    height = top + sum(tile + 54 + prompt_height for prompt_height in row_heights)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(headings):
        lines = wrapped(draw, title, font(19, True), tile)
        for line, text in enumerate(lines):
            draw.text((gap + column * (tile + gap), 8 + 23 * line), text, font=font(19, True), fill="black")
    y = top
    for row, prompt_height in zip(rows, row_heights):
        for line, text in enumerate(wrapped(draw, row["prompt"], font(17), width - 24)):
            draw.text((gap, y + 20 * line), text, font=font(17), fill="black")
        y += prompt_height
        for column, panel in enumerate(row["panels"]):
            path = Path(panel["path"])
            if sha(path) != panel["sha256"]:
                raise ValueError("Selected native image hash changed")
            SOURCES[str(path.relative_to(ROOT))] = panel["sha256"]
            x = gap + column * (tile + gap)
            with Image.open(path) as image:
                image = image.convert("RGB")
                if image.width != image.height:
                    raise ValueError("Expected square native image")
                canvas.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (x, y))
            for line, text in enumerate(wrapped(draw, panel["details"], font(16), tile)):
                draw.text((x, y + tile + 4 + 18 * line), text, font=font(16), fill="black")
        y += tile + 54
    destination = ROOT / "tex/figures" / filename
    canvas.save(destination)
    return sha(destination)


def detector_images():
    root = ROOT / "results/research_workflow/image_dda/20261025T084419997575Z"
    manifest = read_json(root / "manifest.json")
    prompts = read_json(Path(manifest["partitions"]))["evaluation"]
    folders = [root / "test/initial", root / "test/selected",
               ROOT / "results/research_workflow/image_dda/20261025T213338529186Z/evaluated"]
    states = [read_saved(folder / "evaluation.pt") for folder in folders]
    for state in states:
        if len(state["images"]) != 500 or [r["sample_id"] for r in state["images"]] != [r["sample_id"] for r in prompts]:
            raise ValueError("Image test identities differ")
        for key in ["seed", "sampler", "prompts"]:
            if state["config"][key] != states[0]["config"][key]:
                raise ValueError("Image comparisons use different prompts or sampling seeds")
    rows = []
    for index in [0, 250]:
        panels = []
        for state in states:
            sample = state["images"][index]
            panels.append({**sample, "details": (
                f"DDA AI score {state['scores'][index]:.3f}; CLIP {state['clip'][index]:.3f}")})
        rows.append({"position": index, "prompt": f"#{index}: " + prompts[index]["prompt"], "panels": panels})
    checksum = image_grid("detector-sd15-comparison.png", [
        "Initial SD1.5", "DDPO selected: 5,600 images", "DDPO late: 12,800 images"], rows)
    EXAMPLES["detector_images"] = {"rows": rows, "figure_sha256": checksum,
                                  "sampler": states[0]["config"]}
    (ROOT / "tex/generated/detector_image_examples.tex").write_text(r"""
\begin{figure}[p]\centering
\includegraphics[width=\linewidth,height=.75\textheight,keepaspectratio]{figures/detector-sd15-comparison.png}
\caption{AI-detector evasion with unwatermarked Stable Diffusion 1.5 against the fixed, warmed DDA image classifier. Left is initial SD1.5, middle the validation-selected DDPO checkpoint after 5,600 training images, and right the later 12,800-image checkpoint evaluated exploratorily. Rows are fixed held-out positions 0 and 250 from the 500-prompt test cohort, with complete prompts shown above each row. All columns use identical prompts and DDIM sampling settings, with coupled noise seeds 20261025 plus the batch offset, 50 steps and classifier-free guidance 7.5. DDA AI scores below 0.5 count as evasion; the listed CLIP values measure prompt alignment. Selection is independent of these rows' outcomes. Uniform display resizing leaves the native 512-pixel sources unchanged.}
\label{fig:detector-image-comparison}
\end{figure}
""")


def rar_images():
    root = ROOT / "results/research_workflow/image_quality/20261104T153348508006Z/rar"
    pools = read_json(root / "pairs.json")
    methods = ["weight_clean_lora_2500", "output_nfpa_20"]
    quality = {method: {r["index"]: r for r in read_json(root / (method + ".json"))["rows"]}
               for method in methods}
    rows = []
    for index in [0, 256]:
        pairs = {method: pools[method]["rows"][index] for method in methods}
        first = pairs[methods[0]]
        if any(p["index"] != index or p["reference_sha256"] != first["reference_sha256"] for p in pairs.values()):
            raise ValueError("RAR qualitative references or indices differ")
        panels = [{"path": first["reference"], "sha256": first["reference_sha256"],
                   "details": f"Source index {index}; class {first['label']}"}]
        for method in methods:
            pair, metrics = pairs[method], quality[method][index]
            retention = metrics["clip_alignment"] / metrics["reference_clip_alignment"]
            panels.append({"path": pair["candidate"], "sha256": pair["candidate_sha256"],
                           "details": f"LPIPS {metrics['lpips']:.3f}; image CLIP {metrics['clip_image_cosine']:.3f}; class retention {retention:.3f}"})
        rows.append({"position": index, "prompt": f"Source index {index}; ImageNet class index {first['label']}", "panels": panels})
    checksum = image_grid("rar-paired-audit.png", [
        "Initial marked RAR-XL", "Clean LoRA: 2,500 updates", "Output NFPA: strength 20"], rows)
    EXAMPLES["rar_images"] = {"rows": rows, "figure_sha256": checksum}
    (ROOT / "tex/generated/rar_paired_examples.tex").write_text(r"""
\begin{figure}[p]\centering
\includegraphics[width=\linewidth,height=.75\textheight,keepaspectratio]{figures/rar-paired-audit.png}
\caption{DistSeal RAR-XL at 256 pixels, with its original clean decoder. Columns show the initial marked source, a sample after 2,500 clean LoRA updates of the generator, and output-only NFPA purification at strength 20. Rows are fixed source positions 0 and 256 from the 512-image cohort, chosen without outcome filtering. The LoRA column uses the source's class and generation seed; NFPA transforms that same source PNG using an independent SD2.1 model. LPIPS and image CLIP compare each candidate with the left image; class retention is candidate class-text CLIP divided by source class-text CLIP. The retained pair rule requires LPIPS at most 0.2, image CLIP at least 0.9 and class retention at least 0.9. Class-valid replacements can fail the paired-content checks. Uniform resizing is for display only.}
\label{fig:rar-paired-audit}
\end{figure}
""")


def main():
    detector_text()
    watermark_text()
    detector_images()
    rar_images()
    receipt = {
        "selection": "Fixed text position 0 and evenly spaced image positions, without quality or detector-outcome filtering.",
        "sources": SOURCES,
        "examples": EXAMPLES,
        "script_sha256": sha(__file__),
        "display": "Full saved text, with unsupported glyphs and U+2014 represented by explicit Unicode codepoints. Images are uniformly resized for display.",
    }
    (ROOT / "tex/provenance/qualitative_evidence.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print("Saved detector, text-watermark and RAR qualitative comparisons.", flush=True)


if __name__ == "__main__":
    main()
