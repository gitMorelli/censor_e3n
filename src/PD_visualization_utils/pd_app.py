"""
Streamlit front-end.

    streamlit run pd_app.py
    streamlit run pd_app.py -- --table D:/local_copy.parquet   # optional override

Subjects are browsed with the ◀ / ▶ buttons (or the drop-down); the modality,
questionnaire and every display option are kept when moving to the next subject.
"""

from __future__ import annotations

import base64
import io
import sys

import streamlit as st
import streamlit.components.v1 as components

import pd_data as D
import pd_views as V

st.set_page_config(page_title="PD handwriting inspector", layout="wide")

if "--table" in sys.argv:
    D.TABLE_PATH = sys.argv[sys.argv.index("--table") + 1]


@st.cache_data(show_spinner="loading table…")
def table(path: str | None):
    return D.load_table(path)


def sticky_select(container, label, options, state_key, default_index=0, **kw):
    """A selectbox that keeps its value when the option list changes
    (i.e. when you move to another subject)."""
    options = list(options)
    if not options:
        container.caption(f"{label}: —")
        return None
    prev = st.session_state.get(state_key)
    idx = options.index(prev) if prev in options else min(default_index, len(options) - 1)
    val = container.selectbox(label, options, index=idx, **kw)
    st.session_state[state_key] = val
    return val


def show_exact(img, max_height: int = 900):
    """Render at the exact pixel size (scrollable) instead of letting the browser
    fit the image to the column width."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode()
    w, h = img.size
    components.html(
        f'<div style="overflow:auto;max-height:{max_height}px;border:1px solid #ddd;background:#fff">'
        f'<img src="data:image/png;base64,{data}" width="{w}" height="{h}"></div>',
        height=min(h + 30, max_height + 20),
    )
    st.download_button("download PNG", buf.getvalue(), "view.png", "image/png")


# --------------------------------------------------------------------------- #
# data + subject navigation
# --------------------------------------------------------------------------- #
side = st.sidebar
side.header("data")
tbl_path = side.text_input("table", str(D.TABLE_PATH))
share = side.text_input("share root", str(D.SHARE_ROOT))
D.SHARE_ROOT = type(D.SHARE_ROOT)(share)

try:
    df = table(tbl_path)
except Exception as e:  # noqa: BLE001
    st.error(f"could not read the table: {e}")
    st.stop()

side.header("subject")
with side.expander("filter the subject list", expanded=False):
    splits = sorted(map(str, df["split"].dropna().unique())) if "split" in df else []
    keep_split = st.multiselect("split", splits, default=splits)
    who = st.radio("rows", ["all", "cases only", "controls only"], horizontal=True)
    needle = st.text_input("id contains", "")

sub = df
if splits and keep_split:
    sub = sub[sub["split"].astype(str).isin(keep_split)]
if who != "all":
    mask = sub.apply(D.is_case, axis=1)
    sub = sub[mask if who == "cases only" else ~mask]
if needle:
    sub = sub[sub["unique_id"].str.contains(needle, case=False, na=False)]

uids = sub["unique_id"].tolist()
if not uids:
    st.warning("no subject matches the filter")
    st.stop()

st.session_state.setdefault("pos", 0)
st.session_state.pos = min(st.session_state.pos, len(uids) - 1)

nav1, nav2 = side.columns(2)
if nav1.button("◀ prev", use_container_width=True):
    st.session_state.pos = (st.session_state.pos - 1) % len(uids)
if nav2.button("next ▶", use_container_width=True):
    st.session_state.pos = (st.session_state.pos + 1) % len(uids)

chosen = side.selectbox("unique_id", uids, index=st.session_state.pos)
if chosen != uids[st.session_state.pos]:
    st.session_state.pos = uids.index(chosen)
uid = uids[st.session_state.pos]
side.caption(f"{st.session_state.pos + 1} / {len(uids)} in the filtered list")

row = D.subject_row(df, uid)
sid, gid = row["subject_id"], row["group_id"]

side.header("view")
view = sticky_select(
    side, "view",
    ["1 · sequence", "2 · group comparison", "3 · pages + boxes", "4 · pages + extracted"],
    "view",
)
scale = side.slider("scale (1.0 = original pixels)", 0.05, 2.0, 0.35, 0.05)
show_meta = side.checkbox("show subject metadata", False)

avail_q = D.list_available_q(sid)
side.caption(f"subject {sid} · group {gid} · q folders in tar: {avail_q or '—'}")

# --------------------------------------------------------------------------- #
# shared controls
# --------------------------------------------------------------------------- #
if view.startswith(("1", "2")):
    modality = sticky_select(side, "modality", D.all_modalities(sid) or D.MODALITY_ORDER, "modality")
    pattern_col = sticky_select(
        side, "availability pattern",
        ["grid_pattern", "case_grid_pattern", "rempli_pattern", "case_pattern", "(none)"], "pattern",
    )
    pattern_col = None if pattern_col == "(none)" else pattern_col
    mode = sticky_select(
        side, "sequence",
        ["all", "upto_last_avail", "after_last_avail", "last_available", "custom"], "mode",
        format_func=lambda m: {
            "all": "all available",
            "upto_last_avail": "up to last_avail_q",
            "after_last_avail": "after last_avail_q",
            "last_available": "last available only",
            "custom": "custom selection",
        }[m],
    )
    qs = None
    if mode == "custom":
        qs = side.multiselect("questionnaires", list(range(1, D.N_Q + 1)),
                              default=D.select_qs(row, "all", pattern_column=pattern_col))

# --------------------------------------------------------------------------- #
if view.startswith("1"):
    ncols = side.slider("columns", 1, 13, 6)
    st.subheader(f"{uid} — {modality}")
    show_exact(V.view_sequence(df, uid, modality, qs=qs, mode=mode, pattern_column=pattern_col,
                               ncols=ncols, scale=scale, show_metadata=show_meta))

elif view.startswith("2"):
    all_ctrl = D.select_group(df, gid, n_controls=None)
    ctrl_ids = [r["unique_id"] for _, r in all_ctrl.iterrows() if not D.is_case(r)]
    n_ctrl = side.number_input("controls to show", 0, max(len(ctrl_ids), 1), min(2, len(ctrl_ids)))
    pick = side.multiselect(f"or pick specific controls ({len(ctrl_ids)} in the group)", ctrl_ids)
    align = side.checkbox("align columns on the union of questionnaires", True)
    st.subheader(f"group {gid} — {modality}")
    show_exact(V.view_group(df, gid, modality, mode=mode, pattern_column=pattern_col,
                            align_qs=align, scale=scale, show_metadata=show_meta,
                            n_controls=None if pick else int(n_ctrl),
                            control_ids=pick or None))

else:
    q = sticky_select(side, "questionnaire", avail_q or list(range(1, D.N_Q + 1)), "q")
    pages_all = D.list_pages(sid, q)
    pages = side.multiselect("pages", pages_all, default=pages_all)

    side.markdown("**boxes**")
    try:
        det_map = D.load_detections(sid, q)
        classes = sorted({str(d.get("class_name")) for ds in det_map.values() for d in ds})
    except Exception:  # noqa: BLE001
        classes = []
    keep = side.multiselect("classes", classes, default=classes)
    draw_opts = {
        "classes": set(keep) if classes else None,
        "min_conf": side.slider("min confidence", 0.0, 1.0, 0.0, 0.05),
        "show_labels": side.checkbox("class labels", True),
        "show_conf": side.checkbox("confidences", True),
        "draw_text_breaks": side.checkbox("text_breaks separators", True),
        "show_regions": side.checkbox("special regions (partial_boxes_coords.csv)", True),
    }

    if view.startswith("3"):
        ncols = side.slider("columns", 1, 8, 4)
        st.subheader(f"{uid} — Q{q} — pages")
        img = V.view_pages(df, uid, q, pages=pages, draw_opts=draw_opts,
                           ncols=ncols, scale=scale, show_metadata=show_meta)
    else:
        mods_q = D.list_modalities(sid, q)
        chosen_mods = side.multiselect("modalities (2nd row)", mods_q, default=mods_q[:3])
        mscale = side.slider("extra zoom on the extracted row", 1.0, 6.0, 1.0, 0.5)
        st.subheader(f"{uid} — Q{q} — pages + extracted data")
        img = V.view_combined(df, uid, q, chosen_mods, pages=pages, draw_opts=draw_opts,
                              scale=scale, modality_scale=mscale, show_metadata=show_meta)
    show_exact(img)

if show_meta:
    with st.expander("metadata (all columns except case_dt_date* and q_N_num*)", expanded=True):
        st.json({k: str(v) for k, v in D.metadata_dict(row).items()})