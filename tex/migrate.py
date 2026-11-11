"""Regenerate separate LaTeX records from the frozen research-document snapshot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

from PIL import Image


TEX = Path(__file__).resolve().parent
ROOT = TEX.parent


def escaped(text):
    replacements = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
                    "$": r"\$", "&": r"\&", "#": r"\#", "_": r"\_",
                    "%": r"\%", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in text)


def convert(pandoc):
    originals = json.loads((TEX / "provenance/source_documents.json").read_text())
    missing = {int(c, 16) for c in json.loads((TEX / "provenance/missing_glyphs.json").read_text())}
    mapping = {p: "tex/records/" + str(Path(p).with_suffix(".tex")) for p in originals}
    assets = {}

    def asset(path):
        relative = str(path.relative_to(ROOT))
        if path.suffix.lower() == ".webp":
            if relative not in assets:
                output = TEX / "figures" / path.with_suffix(".png").name
                with Image.open(path) as image:
                    image.convert("RGB").save(output)
                assets[relative] = str(output.relative_to(TEX))
            return assets[relative]
        return "../" + relative

    def walk(node, source):
        if isinstance(node, list):
            result = []
            for n in node:
                if isinstance(n, dict) and n.get("t") == "Table" and len(n["c"][2]) > 6:
                    table = n["c"]
                    rows = table[3][1] + table[5][1]
                    for body in table[4]:
                        rows += body[2] + body[3]
                    if all(cell[2:4] == [1, 1] for row in rows for cell in row[1]):
                        shared = 3 if len(table[2]) >= 10 else 1
                        for start in range(shared, len(table[2]), 6 - shared):
                            indices = list(range(shared)) + list(range(start, min(start + 6 - shared, len(table[2]))))
                            panel = copy.deepcopy(n)
                            panel["c"][2] = [panel["c"][2][i] for i in indices]
                            panel_rows = panel["c"][3][1] + panel["c"][5][1]
                            for body in panel["c"][4]:
                                panel_rows += body[2] + body[3]
                            for row in panel_rows:
                                row[1] = [row[1][i] for i in indices]
                            result.append(walk(panel, source))
                        continue
                result.append(walk(n, source))
            return result
        if not isinstance(node, dict):
            return node
        kind = node.get("t")
        prefix = "record-" + re.sub(r"[^a-zA-Z0-9]+", "-", str(source.relative_to(ROOT))) + "-"
        if kind == "Header" and node["c"][1][0]:
            node["c"][1][0] = prefix + node["c"][1][0]
        if kind == "RawBlock" and node["c"][0] == "html":
            blocks = re.findall(r"<pre>(.*?)</pre>", node["c"][1], re.S)
            if blocks:
                return {"t": "CodeBlock", "c": [["", [], []], html.unescape("\n".join(blocks))]}
        if kind == "Code":
            value = node["c"][1]
            return {"t": "RawInline", "c": ["latex", r"\texttt{\seqsplit{" + escaped(value) + "}}"]}
        if kind == "Str" and "_" in node["c"]:
            return {"t": "RawInline", "c": ["latex", escaped(node["c"]).replace(r"\_", r"\_\allowbreak{}") ]}
        if kind == "Table":
            columns = node["c"][2]
            for column in columns:
                column[1] = {"t": "ColWidth", "c": 1 / len(columns)}
        if kind in ("Link", "Image"):
            target = node["c"][-1][0]
            if kind == "Link" and target.startswith("#"):
                node["c"][-1][0] = "#" + prefix + target[1:]
            if not re.match(r"^[a-zA-Z]+:|^#", target):
                path = (source.parent / target.split("#")[0]).resolve()
                if path.is_relative_to(ROOT):
                    relative = str(path.relative_to(ROOT))
                    if kind == "Image" and path.is_file():
                        node["c"][-1][0] = asset(path)
                    elif relative in mapping:
                        node["c"][-1][0] = str(Path(mapping[relative]).relative_to("tex"))
                    else:
                        node["c"][-1][0] = "../" + relative
        return {key: walk(value, source) for key, value in node.items()}

    manifest = {}
    for relative, raw in sorted(originals.items()):
        ast = json.loads(subprocess.check_output(
            [pandoc, "-f", "markdown+tex_math_dollars+tex_math_single_backslash+raw_html", "-t", "json"],
            input=raw.encode()))
        result = subprocess.check_output(
            [pandoc, "-f", "json", "-t", "latex", "--syntax-highlighting=none"],
            input=json.dumps(walk(ast, ROOT / relative)).encode()).decode()
        result = "".join(f"[U+{ord(c):04X}]" if ord(c) in missing or c == chr(8212) else c for c in result)
        result = result.replace(r"\begin{verbatim}", r"\begin{Verbatim}").replace(r"\end{verbatim}", r"\end{Verbatim}")
        result = result.replace(r"\begin{figure}", r"\begin{recordfigure}").replace(r"\end{figure}", r"\end{recordfigure}")
        result = result.replace(r"{\def\LTcaptype{none}", r"{\scriptsize\setlength{\tabcolsep}{3pt}\def\LTcaptype{none}")
        output = ROOT / mapping[relative]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result)
        manifest[relative] = {"source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                              "latex": mapping[relative], "latex_sha256": hashlib.sha256(result.encode()).hexdigest()}
    (TEX / "provenance/migration.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Regenerated {len(manifest)} LaTeX records; source documents and section selection unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pandoc", default="pandoc")
    convert(parser.parse_args().pandoc)
