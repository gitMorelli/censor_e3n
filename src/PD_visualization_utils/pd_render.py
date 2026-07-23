"""
Rendering layer: draws detections on pages and composes several images into a
single view.

Composition rule: every image in a view is resized by the *same* factor, so
aspect ratios and relative sizes are exactly those of the source files.  With
scale=1.0 the pixels are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

BG = (252, 252, 252)
FG = (30, 30, 30)
MUTED = (120, 120, 120)
BREAK_COLOR = (0, 170, 255)
REGION_COLOR = (255, 140, 0)
CASE_BORDER = (200, 30, 30)
CONTROL_BORDER = (40, 90, 200)

PALETTE = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (166, 86, 40), (247, 129, 191), (23, 190, 207),
    (188, 189, 34), (127, 127, 127),
]


@lru_cache(maxsize=16)
def font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def color_for(class_name: str) -> tuple[int, int, int]:
    return PALETTE[sum(map(ord, str(class_name))) % len(PALETTE)]


# --------------------------------------------------------------------------- #
# detections
# --------------------------------------------------------------------------- #
def _xyxy(det: dict) -> tuple[float, float, float, float] | None:
    if det.get("bbox_xyxy"):
        x1, y1, x2, y2 = det["bbox_xyxy"][:4]
        return float(x1), float(y1), float(x2), float(y2)
    if det.get("bbox_xywh"):
        cx, cy, w, h = det["bbox_xywh"][:4]
        return float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)
    return None


def draw_detections(
    image: Image.Image,
    detections: list[dict],
    *,
    classes: set[str] | None = None,
    min_conf: float = 0.0,
    show_labels: bool = True,
    show_conf: bool = True,
    draw_text_breaks: bool = True,
    region_boxes: list[tuple[float, float, float, float]] | None = None,
) -> Image.Image:
    """Overlay detection boxes (and text-break separators) at native resolution.

    Coordinates are taken as-is, i.e. already in the page image's pixel space.
    ``text_breaks`` are treated as offsets from the box's x1 (as in the reference
    snippet); if a value is larger than the box width it is treated as absolute.
    """
    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    lw = max(2, round(min(out.size) / 500))
    lab_font = font(max(12, round(min(out.size) / 70)))

    for box in region_boxes or []:
        d.rectangle([box[0], box[1], box[2], box[3]], outline=REGION_COLOR, width=lw)

    for det in detections or []:
        name = str(det.get("class_name", det.get("class_id", "?")))
        conf = float(det.get("confidence", 1.0) or 0.0)
        if classes is not None and name not in classes:
            continue
        if conf < min_conf:
            continue
        box = _xyxy(det)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        col = color_for(name)
        w = lw + (1 if det.get("is_partial") else 0)
        d.rectangle([x1, y1, x2, y2], outline=col, width=w)

        if draw_text_breaks and det.get("text_breaks"):
            bw = x2 - x1
            for xb in det["text_breaks"]:
                xb = float(xb)
                x = xb if xb > bw else x1 + xb
                if x1 < x < x2:
                    d.line([(x, y1), (x, y2)], fill=BREAK_COLOR, width=lw)

        if show_labels:
            txt = name + (f" {conf:.2f}" if show_conf else "") + (" *" if det.get("is_partial") else "")
            tb = d.textbbox((0, 0), txt, font=lab_font)
            th = tb[3] - tb[1] + 4
            ty = max(0, y1 - th)
            d.rectangle([x1, ty, x1 + (tb[2] - tb[0]) + 6, ty + th], fill=col)
            d.text((x1 + 3, ty + 2), txt, fill=(255, 255, 255), font=lab_font)
    return out


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    image: Image.Image | None = None
    label: str = ""          # drawn above the image
    sublabel: str = ""       # second line above the image (e.g. the date)
    text: str = ""           # text-only cell (row gutter / metadata panel)
    border: tuple[int, int, int] | None = None
    width_hint: int = 300    # width used for text-only cells


LABEL_SIZE = 14
TITLE_SIZE = 20


def _wrap(draw: ImageDraw.ImageDraw, text: str, fnt, max_w: int) -> list[str]:
    lines: list[str] = []
    for para in str(text).split("\n"):
        cur = ""
        for word in para.split(" "):
            trial = (cur + " " + word).strip()
            if draw.textlength(trial, font=fnt) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def compose(
    rows: list[list[Cell]],
    *,
    scale: float = 1.0,
    gap: int = 18,
    pad: int = 20,
    title: str = "",
    footer: str = "",
    bg: tuple[int, int, int] = BG,
) -> Image.Image:
    """Lay cells out on a grid, columns aligned across rows."""
    lab_f, tit_f = font(LABEL_SIZE), font(TITLE_SIZE)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    line_h = LABEL_SIZE + 5

    # 1. scale every image by the same factor
    prepared: list[list[tuple[Cell, Image.Image | None, list[str]]]] = []
    for row in rows:
        prow = []
        for cell in row:
            img = cell.image
            if img is not None and scale != 1.0:
                w, h = img.size
                img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            tlines = _wrap(probe, cell.text, lab_f, cell.width_hint) if cell.text else []
            prow.append((cell, img, tlines))
        prepared.append(prow)

    ncols = max((len(r) for r in prepared), default=1)
    col_w = [0] * ncols
    row_h = [0] * len(prepared)
    for i, prow in enumerate(prepared):
        for j, (cell, img, tlines) in enumerate(prow):
            n_lab = bool(cell.label) + bool(cell.sublabel)
            if img is not None:
                w, h = img.size
            else:
                w = cell.width_hint
                h = max(line_h * len(tlines), 1) if tlines else 40
            col_w[j] = max(col_w[j], w)
            row_h[i] = max(row_h[i], h + n_lab * line_h)

    head_h = (TITLE_SIZE + 12) * (len(_wrap(probe, title, tit_f, 4000)) if title else 0)
    total_w = pad * 2 + sum(col_w) + gap * (ncols - 1)
    foot_lines = _wrap(probe, footer, lab_f, max(400, total_w - 2 * pad)) if footer else []
    foot_h = line_h * len(foot_lines) + (10 if foot_lines else 0)
    total_h = pad * 2 + head_h + sum(row_h) + gap * (len(prepared) - 1) + foot_h

    canvas = Image.new("RGB", (max(total_w, 320), max(total_h, 120)), bg)
    d = ImageDraw.Draw(canvas)

    y = pad
    if title:
        for ln in _wrap(probe, title, tit_f, 4000):
            d.text((pad, y), ln, fill=FG, font=tit_f)
            y += TITLE_SIZE + 12

    for i, prow in enumerate(prepared):
        x = pad
        for j, (cell, img, tlines) in enumerate(prow):
            cy = y
            if cell.label:
                d.text((x, cy), cell.label, fill=FG, font=lab_f)
                cy += line_h
            if cell.sublabel:
                d.text((x, cy), cell.sublabel, fill=MUTED, font=lab_f)
                cy += line_h
            if img is not None:
                canvas.paste(img, (x, cy))
                if cell.border:
                    d.rectangle([x - 3, cy - 3, x + img.size[0] + 2, cy + img.size[1] + 2],
                                outline=cell.border, width=3)
            else:
                for ln in tlines:
                    d.text((x, cy), ln, fill=FG if not cell.label else MUTED, font=lab_f)
                    cy += line_h
            x += col_w[j] + gap
        y += row_h[i] + gap

    if foot_lines:
        y += 4
        for ln in foot_lines:
            d.text((pad, y), ln, fill=MUTED, font=lab_f)
            y += line_h
    return canvas


def missing_cell(label: str, why: str, width: int = 260) -> Cell:
    return Cell(label=label, text=f"— {why} —", width_hint=width)