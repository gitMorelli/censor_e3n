"""
The four inspection views.  Each returns a single PIL image, so they can be used
from the Streamlit app, a notebook (`view.show()`), or the CLI.
"""

from __future__ import annotations

import pandas as pd
from PIL import Image

import pd_data as D
import pd_render as R
from pd_render import Cell


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def subject_title(row: pd.Series) -> str:
    return (
        f"{row['unique_id']}   group {row['group_id']}   "
        f"case_control={row.get('case_control')}   "
        f"diag_park_final1_quest={row.get('diag_park_final1_quest')}   "
        f"last_avail_q={row.get('last_avail_q')}   split={row.get('split')}"
    )


def metadata_footer(row: pd.Series, enabled: bool) -> str:
    if not enabled:
        return ""
    md = D.metadata_dict(row)
    return "   |   ".join(f"{k}={v}" for k, v in md.items())


def border_for(row: pd.Series) -> tuple[int, int, int]:
    return R.CASE_BORDER if str(row.get("case_control")) in ("1", "1.0", "True") else R.CONTROL_BORDER


def _modality_cell(row: pd.Series, q: int, modality: str, *, with_date: bool = True) -> Cell:
    sid = row["subject_id"]
    label = f"Q{q}  {modality}"
    sub = D.date_of(row, q) if with_date else ""
    try:
        img = D.load_modality(sid, q, modality)
    except (FileNotFoundError, KeyError) as e:
        c = R.missing_cell(label, type(e).__name__)
        c.sublabel = sub
        return c
    return Cell(image=img, label=label, sublabel=sub)


def _page_cell(row: pd.Series, q: int, page: int, draw_opts: dict) -> Cell:
    sid = row["subject_id"]
    label = f"Q{q}  page {page}"
    try:
        img = D.load_page(sid, q, page)
    except (FileNotFoundError, KeyError) as e:
        return R.missing_cell(label, type(e).__name__)
    dets = draw_opts.get("_detections", {}).get(page, [])
    regions = draw_opts.get("_regions", {}).get(page, []) if draw_opts.get("show_regions", True) else []
    img = R.draw_detections(
        img,
        dets,
        classes=draw_opts.get("classes"),
        min_conf=draw_opts.get("min_conf", 0.0),
        show_labels=draw_opts.get("show_labels", True),
        show_conf=draw_opts.get("show_conf", True),
        draw_text_breaks=draw_opts.get("draw_text_breaks", True),
        region_boxes=regions,
    )
    sub = f"{len(dets)} det." + (f" · {len(regions)} region" if regions else "")
    return Cell(image=img, label=label, sublabel=sub)


def _load_overlays(sid: str, q: int, draw_opts: dict) -> dict:
    opts = dict(draw_opts)
    try:
        opts["_detections"] = D.load_detections(sid, q)
    except (FileNotFoundError, KeyError):
        opts["_detections"] = {}
    opts["_regions"] = D.load_region_boxes(sid, q) if draw_opts.get("show_regions", True) else {}
    return opts


# --------------------------------------------------------------------------- #
# 1. sequence of one modality for one subject
# --------------------------------------------------------------------------- #
def view_sequence(
    df: pd.DataFrame,
    unique_id: str,
    modality: str,
    *,
    qs: list[int] | None = None,
    mode: str = "all",
    pattern_column: str | None = "grid_pattern",
    ncols: int = 6,
    scale: float = 1.0,
    show_metadata: bool = False,
) -> Image.Image:
    row = D.subject_row(df, unique_id)
    qs = qs if qs is not None else D.select_qs(row, mode, pattern_column=pattern_column)
    cells = [_modality_cell(row, q, modality) for q in qs]
    rows = [cells[i:i + ncols] for i in range(0, len(cells), ncols)] or [[R.missing_cell("", "no questionnaire selected")]]
    return R.compose(
        rows,
        scale=scale,
        title=f"[{modality}]  " + subject_title(row),
        footer=metadata_footer(row, show_metadata),
    )


# --------------------------------------------------------------------------- #
# 2. same modality across the subjects of one case/control group
# --------------------------------------------------------------------------- #
def view_group(
    df: pd.DataFrame,
    group_id: str,
    modality: str,
    *,
    mode: str = "all",
    pattern_column: str | None = "grid_pattern",
    align_qs: bool = True,
    scale: float = 1.0,
    show_metadata: bool = False,
    n_controls: int | None = 2,
    control_ids: list[str] | None = None,
) -> Image.Image:
    members = D.select_group(df, str(group_id), n_controls=n_controls, control_ids=control_ids)
    if members.empty:
        return R.compose([[R.missing_cell("", f"group {group_id} not found")]])

    per_subject = {r["unique_id"]: D.select_qs(r, mode, pattern_column=pattern_column)
                   for _, r in members.iterrows()}
    union = sorted({q for qs in per_subject.values() for q in qs})

    rows: list[list[Cell]] = []
    for _, row in members.iterrows():
        gutter = Cell(
            label=row["unique_id"],
            text=(f"case_control={row.get('case_control')}\n"
                  f"diag={row.get('diag_park_final1_quest')}\n"
                  f"last_avail_q={row.get('last_avail_q')}\n"
                  f"split={row.get('split')}"),
            width_hint=180,
        )
        qs = union if align_qs else per_subject[row["unique_id"]]
        cells = [_modality_cell(row, q, modality) for q in qs]
        for c in cells:
            c.border = border_for(row)
        rows.append([gutter] + cells)

    footer = ""
    if show_metadata:
        footer = "\n".join(metadata_footer(r, True) for _, r in members.iterrows())
    return R.compose(
        rows,
        scale=scale,
        title=(f"group {group_id}  |  modality [{modality}]  |  "
               f"{int(members.apply(D.is_case, axis=1).sum())} case(s) + "
               f"{int((~members.apply(D.is_case, axis=1)).sum())} control(s) shown"),
        footer=footer,
    )


# --------------------------------------------------------------------------- #
# 3. censored pages of one questionnaire with the detection boxes
# --------------------------------------------------------------------------- #
def view_pages(
    df: pd.DataFrame,
    unique_id: str,
    q: int,
    *,
    pages: list[int] | None = None,
    draw_opts: dict | None = None,
    ncols: int = 4,
    scale: float = 1.0,
    show_metadata: bool = False,
) -> Image.Image:
    row = D.subject_row(df, unique_id)
    sid = row["subject_id"]
    opts = _load_overlays(sid, q, draw_opts or {})
    pages = pages if pages is not None else D.list_pages(sid, q)
    cells = [_page_cell(row, q, p, opts) for p in pages]
    rows = [cells[i:i + ncols] for i in range(0, len(cells), ncols)] or [[R.missing_cell(f"Q{q}", "no page found")]]
    return R.compose(
        rows,
        scale=scale,
        title=f"Q{q} ({D.date_of(row, q)})  censored pages  |  " + subject_title(row),
        footer=metadata_footer(row, show_metadata),
    )


# --------------------------------------------------------------------------- #
# 4. pages + extracted modalities for the same questionnaire
# --------------------------------------------------------------------------- #
def view_combined(
    df: pd.DataFrame,
    unique_id: str,
    q: int,
    modalities: list[str],
    *,
    pages: list[int] | None = None,
    draw_opts: dict | None = None,
    scale: float = 1.0,
    modality_scale: float | None = None,
    show_metadata: bool = False,
) -> Image.Image:
    """Row 1: the N censored pages with boxes.  Row 2: the 1 image per modality.

    Both rows use the same scale factor, so the extracted crops are shown at the
    size they have relative to the pages they were cut from.  `modality_scale`
    can up-scale only the second row when the crops are too small to inspect.
    """
    row = D.subject_row(df, unique_id)
    sid = row["subject_id"]
    opts = _load_overlays(sid, q, draw_opts or {})
    pages = pages if pages is not None else D.list_pages(sid, q)
    modalities = modalities or D.list_modalities(sid, q)

    page_row = [_page_cell(row, q, p, opts) for p in pages] or [R.missing_cell(f"Q{q}", "no page found")]
    mod_row: list[Cell] = []
    for m in modalities:
        c = _modality_cell(row, q, m, with_date=False)
        if c.image is not None and modality_scale and modality_scale != 1.0:
            w, h = c.image.size
            c.image = c.image.resize((max(1, round(w * modality_scale)), max(1, round(h * modality_scale))),
                                     Image.LANCZOS)
            c.sublabel = f"x{modality_scale:g}"
        mod_row.append(c)
    mod_row = mod_row or [R.missing_cell(f"Q{q}", "no extracted modality")]

    return R.compose(
        [page_row, mod_row],
        scale=scale,
        title=f"Q{q} ({D.date_of(row, q)})  pages + extracted  |  " + subject_title(row),
        footer=metadata_footer(row, show_metadata),
    )