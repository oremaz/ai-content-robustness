"""Build tables and a fixed-position grid for the failed survival-filter pilot."""

import hashlib
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def main():
    receipt = json.loads((ROOT / "tex/provenance/survival_pilot.json").read_text())
    directory = Path(receipt["manifest"]).parent
    sources = {}

    def read(path):
        data = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    result = read(directory / "results.json")
    if sources[str((directory / "results.json").relative_to(ROOT))] != receipt["result_sha256"]:
        raise ValueError("Pilot results changed")
    methods = ["original", "inverse", "survival1", "survival4"]
    values = {}
    for method in methods:
        samples = read(directory / method / "samples.json")["rows"]
        quality = read(directory / method / "quality.json")
        if [r["index"] for r in samples] != list(range(32)) or [r["index"] for r in quality] != list(range(32)):
            raise ValueError("Pilot sample/quality alignment changed")
        for row in samples:
            path = directory / method / f"{row['index']:05}.png"
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Pilot image changed")
        values[method] = {"samples": samples, "quality": quality,
                          "summary": {**result["conditions"][method],
                                      "mean_lpips": statistics.mean(r["lpips"] for r in quality),
                                      "mean_clip_image_cosine": statistics.mean(r["clip_image_cosine"] for r in quality)}}
    lines = [r"\begin{table}[ht]\centering\small",
             r"\begin{tabular}{lrrrr}\toprule",
             r"Candidate & Detected & Preservation pass & Mean bit acc. & LPIPS\\\midrule"]
    for method, value in values.items():
        row = value["summary"]
        lines.append(f"{method} & {row['detected']}/32 & {row['preservation_pass']}/32 & "
                     f"{row['mean_bit_accuracy']:.4f} & {row['mean_lpips']:.4f} \\\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{New DC-AE development pilot, before any removal attack. All filtered candidates fail the initial detection and paired-preservation screen. No robustness comparison is performed on them.}\end{table}"]
    (ROOT / "tex/generated/survival_pilot.tex").write_text("\n".join(lines) + "\n")
    positions = [0, 8, 16, 24]
    tile, top = 384, 48
    grid = Image.new("RGB", (4 * tile, 4 * (tile + top)), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype("DejaVuSans.ttf", 17)
    for y, index in enumerate(positions):
        for x, method in enumerate(methods):
            sample = values[method]["samples"][index]
            quality = values[method]["quality"][index]
            if sample["label"] != values["original"]["samples"][index]["label"]:
                raise ValueError("Qualitative class alignment changed")
            with Image.open(directory / method / f"{index:05}.png") as image:
                grid.paste(image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS),
                           (x * tile, y * (tile + top) + top))
            text = f"{method}: idx {index}, class {sample['label']}\nbit acc. {sample['bit_accuracy']:.3f}; LPIPS {quality['lpips']:.3f}"
            draw.text((x * tile + 6, y * (tile + top) + 3), text, font=font, fill="black")
    grid.save(ROOT / "tex/figures/survival-pilot.png")
    save = {"status": receipt["conclusion"], "sources": sources,
            "qualitative_positions": positions, "position_selection": "Every eighth position after the run; no filtering by detection or quality and no preregistration claim.",
            "conditions": {key: value["summary"] for key, value in values.items()}}
    (ROOT / "tex/provenance/survival_pilot_analysis.json").write_text(json.dumps(save, indent=2) + "\n")
    print(json.dumps(save["conditions"], indent=2))


if __name__ == "__main__":
    main()
