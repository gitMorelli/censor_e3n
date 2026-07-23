"""
Data access layer for the PD handwriting inspector.

Everything is addressed by (subject_id, questionnaire, page); the only things that
have to be configured are the two roots below.

    <TABLE_PATH>                                   the parquet table
    <SHARE_ROOT>/censored_files/Q{q}/images/{sid}.tar
        *{n}.png (page images, identified by the trailing number only),
        partial_boxes_coords.csv (special-region boxes)
    <SHARE_ROOT>/extracted_data/grids/data/id_{sid}.tar
        q{q}/detections.json, q{q}/{modality}.png

Both can be overridden with the PD_TABLE / PD_SHARE environment variables, or by
assigning to pd_data.TABLE_PATH / pd_data.SHARE_ROOT before the first access.
"""

from __future__ import annotations

import io
import json
import os
import re
import tarfile
import threading
from functools import lru_cache
from pathlib import Path

import pandas as pd
from PIL import Image

N_Q = 13

DEFAULT_TABLE = r"\\vms-e34n-databr\2025-handwriting\visualization_of_pd_samples\PD_training_set_20_07_26.parquet"
DEFAULT_SHARE = r"\\smb-recherche-s1.prod-powerscale.intra.igr.fr\E34N_HANDWRITING$"

TABLE_PATH = Path(os.environ.get("PD_TABLE", DEFAULT_TABLE))
SHARE_ROOT = Path(os.environ.get("PD_SHARE", DEFAULT_SHARE))

# canonical display order of the extracted modalities; anything else found in the
# tar is appended alphabetically, so new modalities show up without code changes.
MODALITY_ORDER = [
    "hand",
    "hand_partial_full",
    "hand_sentences_full",
    "number",
    "number_random",
    "X",
    "X_random",
]

# page images are identified by the trailing number of the file name only, whatever
# the prefix is (page_3.png, censored_page_w_3.png, ... all mean "page 3")
PAGE_RX = re.compile(r"^.*?(\d+)\.png$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def censored_tar_path(subject_id: str, q: int) -> Path:
    return SHARE_ROOT / "censored_files" / f"Q{q}" / "images" / f"{subject_id}.tar"


def grid_tar_path(subject_id: str) -> Path:
    return SHARE_ROOT / "extracted_data" / "grids" / "data" / f"id_{subject_id}.tar"


def split_uid(unique_id: str) -> tuple[str, str]:
    """'XXXX_YYYY' -> (subject_id, group_id).  Kept as strings (leading zeros)."""
    sid, _, gid = str(unique_id).partition("_")
    return sid, gid


# --------------------------------------------------------------------------- #
# tar access
# --------------------------------------------------------------------------- #
class TarArchive:
    """Random access to a tar, indexed case-insensitively and prefix-tolerantly.

    Members may or may not be nested under a top-level folder (``id_1234/q1/...``
    vs ``q1/...``); lookups are done by path suffix so both work.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"tar not found: {self.path}")
        self._lock = threading.Lock()
        self._tf = tarfile.open(self.path, "r")
        self._index: dict[str, tarfile.TarInfo] = {}
        for m in self._tf.getmembers():
            if m.isfile():
                self._index[self._norm(m.name)] = m

    @staticmethod
    def _norm(name: str) -> str:
        name = name.replace("\\", "/")
        while name.startswith("./"):
            name = name[2:]
        return name.lstrip("/").lower()

    # -- lookup ------------------------------------------------------------- #
    def find(self, rel: str) -> str | None:
        key = self._norm(rel)
        if key in self._index:
            return key
        for k in self._index:
            if k.endswith("/" + key):
                return k
        return None

    def has(self, rel: str) -> bool:
        return self.find(rel) is not None

    def match(self, pattern: str) -> list[tarfile.TarInfo]:
        """All members whose normalised name matches `pattern` (case-insensitive)."""
        rx = re.compile(pattern, re.IGNORECASE)
        return [m for k, m in self._index.items() if rx.search(k)]

    def read(self, rel: str) -> bytes:
        key = self.find(rel)
        if key is None:
            raise KeyError(f"{rel} not in {self.path.name}")
        with self._lock:
            f = self._tf.extractfile(self._index[key])
            if f is None:
                raise KeyError(f"{rel} is not a regular file in {self.path.name}")
            return f.read()

    def image(self, rel: str) -> Image.Image:
        img = Image.open(io.BytesIO(self.read(rel)))
        img.load()
        return img.convert("RGB")

    def close(self) -> None:
        with self._lock:
            self._tf.close()


@lru_cache(maxsize=32)
def grid_archive(subject_id: str) -> TarArchive:
    return TarArchive(grid_tar_path(subject_id))


@lru_cache(maxsize=64)
def censored_archive(subject_id: str, q: int) -> TarArchive:
    return TarArchive(censored_tar_path(subject_id, q))


def clear_caches() -> None:
    for fn in (grid_archive, censored_archive, load_table):
        fn.cache_clear()


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def load_table(path: str | None = None) -> pd.DataFrame:
    p = Path(path) if path else TABLE_PATH
    df = pd.read_csv(p) if p.suffix.lower() in (".csv", ".tsv") else pd.read_parquet(p)
    df["unique_id"] = df["unique_id"].astype(str)
    df["subject_id"] = df["unique_id"].map(lambda u: split_uid(u)[0])
    df["group_id"] = df["unique_id"].map(lambda u: split_uid(u)[1])
    return df


def subject_row(df: pd.DataFrame, unique_id: str) -> pd.Series:
    hit = df.loc[df["unique_id"] == str(unique_id)]
    if hit.empty:  # allow passing the bare subject id
        hit = df.loc[df["subject_id"] == str(unique_id)]
    if hit.empty:
        raise KeyError(f"{unique_id} not found in the table")
    return hit.iloc[0]


def group_members(df: pd.DataFrame, group_id: str) -> pd.DataFrame:
    g = df.loc[df["group_id"] == str(group_id)].copy()
    # case first, then by id, so the case row is always on top of a comparison
    g["_case"] = pd.to_numeric(g.get("case_control"), errors="coerce").fillna(0)
    return g.sort_values(["_case", "subject_id"], ascending=[False, True]).drop(columns="_case")


def is_case(row: pd.Series) -> bool:
    return str(row.get("case_control")) in ("1", "1.0", "True", "true")


def select_group(
    df: pd.DataFrame,
    group_id: str,
    *,
    n_controls: int | None = 2,
    control_ids: list[str] | None = None,
) -> pd.DataFrame:
    """All cases of the group + the requested controls (first `n_controls` by id,
    or exactly `control_ids` when given).  n_controls=None -> every control."""
    g = group_members(df, group_id)
    cases = g[g.apply(is_case, axis=1)]
    controls = g[~g.apply(is_case, axis=1)]
    if control_ids:
        controls = controls[controls["unique_id"].isin([str(c) for c in control_ids])]
    elif n_controls is not None:
        controls = controls.iloc[:max(0, int(n_controls))]
    return pd.concat([cases, controls])


META_EXCLUDE = re.compile(r"^(case_dt_date|q_\d+_num)")


def metadata_dict(row: pd.Series, exclude: re.Pattern = META_EXCLUDE) -> dict:
    return {k: v for k, v in row.items() if not exclude.match(str(k)) and not str(k).startswith("_")}


def date_of(row: pd.Series, q: int) -> str:
    v = row.get(f"case_dt_dateq{q}")
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return "-"
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d")
    return str(v)[:19]


# --------------------------------------------------------------------------- #
# questionnaire selection
# --------------------------------------------------------------------------- #
def pattern_qs(row: pd.Series, column: str = "grid_pattern") -> list[int]:
    """Questionnaire numbers flagged as present in a 13-char 1/0 pattern column."""
    pat = row.get(column)
    if pat is None or (isinstance(pat, float) and pd.isna(pat)):
        return list(range(1, N_Q + 1))
    pat = str(pat)
    return [i + 1 for i, c in enumerate(pat[:N_Q]) if c == "1"]


SEQUENCE_MODES = ("all", "upto_last_avail", "after_last_avail", "last_available", "custom")


def select_qs(
    row: pd.Series,
    mode: str = "all",
    *,
    pattern_column: str | None = "grid_pattern",
    inclusive: bool = True,
    custom: list[int] | None = None,
) -> list[int]:
    """Questionnaires to display.

    ``upto_last_avail``   q <= last_avail_q  (q < if inclusive=False)
    ``after_last_avail``  q >  last_avail_q  (q >= if inclusive=False)
    ``last_available``    the highest available questionnaire
    Always intersected with `pattern_column` when that column is given.
    """
    avail = pattern_qs(row, pattern_column) if pattern_column else list(range(1, N_Q + 1))
    last = pd.to_numeric(pd.Series([row.get("last_avail_q")]), errors="coerce").iloc[0]

    if mode == "custom":
        qs = [q for q in (custom or []) if 1 <= q <= N_Q]
    elif mode == "last_available":
        qs = [max(avail)] if avail else []
    elif mode in ("upto_last_avail", "after_last_avail"):
        if pd.isna(last):  # e.g. controls without a diagnosis date -> no cut
            qs = list(avail)
        elif mode == "upto_last_avail":
            qs = [q for q in avail if (q <= last if inclusive else q < last)]
        else:
            qs = [q for q in avail if (q > last if inclusive else q >= last)]
    else:
        qs = list(avail)
    return sorted(set(qs))


# --------------------------------------------------------------------------- #
# extracted data (grids tar)
# --------------------------------------------------------------------------- #
def list_available_q(subject_id: str) -> list[int]:
    """Questionnaire folders actually present in the extracted-data tar."""
    try:
        arch = grid_archive(subject_id)
    except FileNotFoundError:
        return []
    qs = set()
    for m in arch.match(r"(?:^|/)q(\d+)/"):
        mm = re.search(r"(?:^|/)q(\d+)/", m.name.replace("\\", "/"), re.IGNORECASE)
        if mm:
            qs.add(int(mm.group(1)))
    return sorted(qs)


def list_modalities(subject_id: str, q: int) -> list[str]:
    """Modality stems present in q{q}/ (original case preserved)."""
    try:
        arch = grid_archive(subject_id)
    except FileNotFoundError:
        return []
    stems = []
    for m in arch.match(rf"(?:^|/)q{q}/[^/]+\.png$"):
        stems.append(Path(m.name.replace("\\", "/")).stem)
    order = {name: i for i, name in enumerate(MODALITY_ORDER)}
    return sorted(set(stems), key=lambda s: (order.get(s, len(order)), s))


def all_modalities(subject_id: str) -> list[str]:
    stems: set[str] = set()
    for q in list_available_q(subject_id):
        stems.update(list_modalities(subject_id, q))
    order = {name: i for i, name in enumerate(MODALITY_ORDER)}
    return sorted(stems, key=lambda s: (order.get(s, len(order)), s))


def load_modality(subject_id: str, q: int, modality: str) -> Image.Image:
    return grid_archive(subject_id).image(f"q{q}/{modality}.png")


def load_detections(subject_id: str, q: int) -> dict[int, list[dict]]:
    """{page_number: [detection, ...]} from q{q}/detections.json."""
    raw = json.loads(grid_archive(subject_id).read(f"q{q}/detections.json"))
    if isinstance(raw, dict):
        raw = [raw]
    out: dict[int, list[dict]] = {}
    for entry in raw:
        page = int(entry.get("page_number", 0))
        out.setdefault(page, []).extend(entry.get("detections", []) or [])
    return out


# --------------------------------------------------------------------------- #
# censored pages (censored tar)
# --------------------------------------------------------------------------- #
def page_files(subject_id: str, q: int) -> dict[int, list[str]]:
    """{page number: [member names]} for every .png ending in a number."""
    try:
        arch = censored_archive(subject_id, q)
    except FileNotFoundError:
        return {}
    out: dict[int, list[str]] = {}
    for m in arch.match(r"\d+\.png$"):
        name = m.name.replace("\\", "/").rsplit("/", 1)[-1]
        mm = PAGE_RX.match(name)
        if mm:
            out.setdefault(int(mm.group(1)), []).append(name)
    return {k: sorted(v) for k, v in sorted(out.items())}


def list_pages(subject_id: str, q: int) -> list[int]:
    return sorted(page_files(subject_id, q))


def load_page(subject_id: str, q: int, page: int) -> Image.Image:
    names = page_files(subject_id, q).get(int(page)) or []
    if not names:
        raise KeyError(f"no page {page} in {censored_tar_path(subject_id, q).name}")
    return censored_archive(subject_id, q).image(names[0])


def load_region_boxes(subject_id: str, q: int) -> dict[int, list[tuple[float, float, float, float]]]:
    """partial_boxes_coords.csv (special regions) -> {page: [(xtl, ytl, xbr, ybr), ...]}."""
    try:
        arch = censored_archive(subject_id, q)
        raw = arch.read("partial_boxes_coords.csv")
    except (FileNotFoundError, KeyError):
        return {}
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip().lower() for c in df.columns]
    out: dict[int, list[tuple[float, float, float, float]]] = {}
    for _, r in df.iterrows():
        m = re.search(r"(\d+)", str(r.get("filename", "")))
        if not m:
            continue
        out.setdefault(int(m.group(1)), []).append(
            (float(r["xtl"]), float(r["ytl"]), float(r["xbr"]), float(r["ybr"]))
        )
    return out