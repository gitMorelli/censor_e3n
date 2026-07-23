"""
Headless use, for when Streamlit is not an option.

    python pd_cli.py sequence --id 1234_56 --modality hand --mode upto_last_avail
    python pd_cli.py group    --id 1234_56 --modality number --scale 0.4
    python pd_cli.py pages    --id 1234_56 --q 5
    python pd_cli.py combined --id 1234_56 --q 5 --modality hand --modality number
    ... add  --out view.png  to save instead of opening a window.
"""

from __future__ import annotations

import argparse

import src.PD_visualization_utils.pd_data as D
import src.PD_visualization_utils.pd_views as V


def main() -> None:
    p = argparse.ArgumentParser(description="PD handwriting inspector")
    p.add_argument("view", choices=["sequence", "group", "pages", "combined"])
    p.add_argument("--id", required=True, help="unique_id (XXXX_YYYY) or bare subject id")
    p.add_argument("--table")
    p.add_argument("--share")
    p.add_argument("--modality", action="append", default=[])
    p.add_argument("--q", type=int)
    p.add_argument("--qs", type=int, nargs="*")
    p.add_argument("--pages", type=int, nargs="*")
    p.add_argument("--mode", default="all", choices=list(D.SEQUENCE_MODES))
    p.add_argument("--pattern", default="grid_pattern")
    p.add_argument("--ncols", type=int, default=6)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--modality-scale", type=float, default=1.0)
    p.add_argument("--metadata", action="store_true")
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--no-breaks", action="store_true")
    p.add_argument("--no-regions", action="store_true", help="hide partial_boxes_coords.csv regions")
    p.add_argument("--n-controls", type=int, default=2, help="-1 = all controls of the group")
    p.add_argument("--out")
    a = p.parse_args()

    if a.table:
        D.TABLE_PATH = a.table
    if a.share:
        D.SHARE_ROOT = type(D.SHARE_ROOT)(a.share)

    df = D.load_table(a.table)
    pattern = None if a.pattern in ("none", "") else a.pattern
    draw_opts = {
        "show_labels": not a.no_labels,
        "show_conf": not a.no_labels,
        "draw_text_breaks": not a.no_breaks,
        "show_regions": not a.no_regions,
    }

    if a.view == "sequence":
        img = V.view_sequence(df, a.id, a.modality[0] if a.modality else "hand", qs=a.qs,
                              mode=a.mode, pattern_column=pattern, ncols=a.ncols,
                              scale=a.scale, show_metadata=a.metadata)
    elif a.view == "group":
        gid = D.subject_row(df, a.id)["group_id"]
        img = V.view_group(df, gid, a.modality[0] if a.modality else "hand", mode=a.mode,
                           pattern_column=pattern, scale=a.scale, show_metadata=a.metadata,
                           n_controls=None if a.n_controls < 0 else a.n_controls)
    elif a.view == "pages":
        img = V.view_pages(df, a.id, a.q, pages=a.pages, draw_opts=draw_opts,
                           ncols=min(a.ncols, 4), scale=a.scale, show_metadata=a.metadata)
    else:
        img = V.view_combined(df, a.id, a.q, a.modality, pages=a.pages, draw_opts=draw_opts, scale=a.scale, modality_scale=a.modality_scale,
                              show_metadata=a.metadata)

    if a.out:
        img.save(a.out)
        print(f"{a.out}  {img.size[0]}x{img.size[1]}")
    else:
        img.show()


if __name__ == "__main__":
    main()