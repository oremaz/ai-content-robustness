"""Faithful contact sheets and portable documentation checks."""

import hashlib
import html
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
import re
from urllib.parse import unquote

from PIL import Image, ImageDraw, ImageFont


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_positions(count, pairs=8):
    if count < pairs or pairs < 1:
        raise ValueError("Insufficient records for the declared qualitative sample")
    return [i*count//pairs for i in range(pairs)]


def full_text(text):
    return "<pre>"+html.escape(text, quote=False).replace("\u2014", "&#8212;")+"</pre>"


@lru_cache(maxsize=32)
def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        spec = importlib.util.find_spec("matplotlib")
        if spec is None:
            raise RuntimeError("DejaVuSans is required to render reproducible report figures")
        return ImageFont.truetype(str(Path(spec.origin).parent/"mpl-data/fonts/ttf"/name), size)


def wrapped(draw, text, face, width):
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            trial = (line+" "+word).strip()
            if line and draw.textlength(trial, font=face) > width:
                lines.append(line);line = word
            else:
                line = trial
        lines.append(line)
    return lines


def draw_lines(draw, xy, lines, face, height, fill="#202b39"):
    x, y = xy
    for line in lines:
        draw.text((x,y),line,font=face,fill=fill);y += height


def image_grid(path, title, pairs, panel_size=256):
    if len(pairs) != 8:
        raise ValueError("Published image grids require exactly eight pairs")
    gap, top, header = 12, 90, 75
    cell = panel_size+gap
    row = panel_size+header+gap
    canvas = Image.new("RGB", (4*cell+gap, 4*row+top+gap), "#f4f6f9")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap,12),title,font=font(20,True),fill="#17263b")
    draw.text((gap,44),"8 fixed pairs: original / after, original / after. Display resize only.",font=font(14),fill="#425269")
    for i, pair in enumerate(pairs):
        for j, side in enumerate(("before","after")):
            x = gap+((i%2)*2+j)*cell;y = top+(i//2)*row
            image = pair[side]
            with Image.open(image) if isinstance(image,(str,Path)) else image.copy() as source:
                picture = source.convert("RGB").resize((panel_size,panel_size),Image.Resampling.LANCZOS)
            draw.rectangle((x,y,x+panel_size,y+header+panel_size),fill="white")
            label = f"#{pair['index']:03} | "+("Watermarked" if j==0 else "After / control")
            draw.text((x+6,y+5),label,font=font(15,True),fill="#17263b")
            details = wrapped(draw,pair[side+"_label"],font(12),panel_size-12)
            draw_lines(draw,(x+6,y+28),details[:3],font(12),14)
            canvas.paste(picture,(x,y+header))
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path,"WEBP",lossless=True,method=4)


def text_grid(path, title, pairs, preview_chars=360):
    if len(pairs) != 8:
        raise ValueError("Published text grids require exactly eight pairs")
    gap, top, width, height = 14, 100, 390, 390
    canvas = Image.new("RGB",(4*(width+gap)+gap,4*(height+gap)+top),"#f4f6f9")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap,12),title,font=font(22,True),fill="#17263b")
    draw.text((gap,45),f"8 fixed pairs. Preview: up to {preview_chars} original characters; full verbatim texts are on the accompanying page.",font=font(14),fill="#425269")
    draw.text((gap,68),"Emoji are shown as Unicode codepoints in previews; full text retains the original symbols.",font=font(12),fill="#425269")
    for i,pair in enumerate(pairs):
        for j,side in enumerate(("before","after")):
            x=gap+((i%2)*2+j)*(width+gap);y=top+(i//2)*(height+gap)
            draw.rectangle((x,y,x+width,y+height),fill="white")
            draw.text((x+9,y+8),f"Example {i+1}: "+("Original" if j==0 else "After / control"),font=font(16,True),fill="#17263b")
            labels=wrapped(draw,pair[side+"_label"],font(12),width-18)
            draw_lines(draw,(x+9,y+35),labels[:3],font(12),16)
            text=pair[side];preview=text[:preview_chars]+(" [...]" if len(text)>preview_chars else "")
            preview="".join(f"[U+{ord(ch):X}]" if ord(ch)>65535 else ch for ch in preview)
            lines=wrapped(draw,preview,font(14),width-18)
            maximum=(height-100)//19
            if len(lines)>maximum:
                lines=lines[:maximum];lines[-1]+=" [...]"
            draw_lines(draw,(x+9,y+95),lines,font(14),19)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path,"WEBP",lossless=True,method=4)


def validate_publication(root):
    root=Path(root)
    manifest=json.loads((root/"publication.json").read_text())
    for name,checksum in manifest["files"].items():
        path=root/name
        if not path.is_file() or sha(path)!=checksum:
            raise ValueError(f"Published artifact changed: {name}")
    for path in root.rglob("*.md"):
        text=re.sub(r"<pre>.*?</pre>","",path.read_text(),flags=re.S)
        if "/lustre/" in text or "../results/" in text:
            raise ValueError(f"Nonportable path in {path}")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",text):
            if target.startswith(("https://","http://","#","mailto:")):
                continue
            target=unquote(target.split("#",1)[0].strip("<>"))
            if Path(target).is_absolute():
                raise ValueError(f"Nonportable absolute link: {path}: {target}")
            if not (path.parent/target).resolve().is_file():
                raise ValueError(f"Broken published link: {path}: {target}")
    for gallery in manifest["galleries"]:
        if len(gallery["sample_ids"])!=8 or len(set(gallery["sample_ids"]))!=8 or gallery["panels"]!=16:
            raise ValueError("Qualitative sample coverage changed")
        with Image.open(root/gallery["asset"]) as image:
            image.verify()
    return {"files":len(manifest["files"]),"galleries":len(manifest["galleries"]),"pairs_per_gallery":8}
