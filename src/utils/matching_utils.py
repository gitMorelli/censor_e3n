import cv2
import numpy as np
from scipy.fftpack import dct
from scipy.optimize import linear_sum_assignment
import re
import unicodedata
from difflib import SequenceMatcher


def check_matching_correspondence(page_dict, pages_list):
    non_corresponding_subset=[]
    for img_id in pages_list:
        if page_dict[img_id]['match_phash']!=page_dict[img_id]['matched_page']:
            non_corresponding_subset.append(img_id)
    return non_corresponding_subset


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    return int(np.count_nonzero(hash1 ^ hash2))

def match_pages_phash(page_dict, template_dict, pages_list, templates_to_consider, gap_threshold=5, max_dist=18):
    #assume phash are already computed

    hashes_pages = []
    hashes_templates = []

    #length of pages and templates may be different if templates only has the ones to censor
    for img_id in pages_list:
        hashes_pages.append(page_dict[img_id]['page_phash'])
    for t_id in templates_to_consider:
        hashes_templates.append(template_dict[t_id]['page_phash'])

    # Cost matrix: Hamming distances
    cost = np.zeros((len(hashes_pages), len(hashes_templates)), dtype=np.int32)
    for i in range(len(hashes_pages)):
        for j in range(len(hashes_templates)):
            cost[i, j] = hamming_distance(hashes_pages[i], hashes_templates[j])

    # Hungarian assignment (minimize total cost)
    assignement = linear_sum_assignment(cost)
    row_ind, col_ind = assignement

    '''if the dimension of the two lists is different -> Every single template will be assigned to a page.
    The function will pick the subset of pages that results in the lowest total Hamming distance.
    The "leftover" pages will be ignored. Because linear_sum_assignment only returns a match for the smaller dimension, 
    the row_ind and col_ind arrays will only have a length equal to the number of templates'''
    
    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append({
            "page_index": pages_list[i],
            "template_index": templates_to_consider[j],
            "hamming": int(cost[i, j]),
        })
    
    confident,report = assignment_confidence_phash(cost, assignement, gap_threshold=gap_threshold, max_dist=max_dist)

    matches_sorted = sorted(matches, key=lambda x: x["page_index"])

    return matches_sorted, cost, confident, report

def update_phash_matches(matches_sorted,page_dict):
    for match in matches_sorted:
        img_id = match['page_index']
        page_dict[img_id]['match_phash']=match['template_index']
    return page_dict


def hungarian_min_cost(cost: np.ndarray):
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def assignment_confidence_phash(cost: np.ndarray, assignment, gap_threshold: int, max_dist: int):
    """
    Returns (is_confident, report_dict).
    Confidence logic:
      - per-row gap test: (2nd best - best) >= gap_threshold
      - matched distance <= max_dist
    """
    n = cost.shape[0]
    row_ind, col_ind = assignment
    row_to_col = {r: c for r, c in zip(row_ind, col_ind)}

    per_row = []
    confident = True

    for r in range(n):
        row = cost[r].astype(int)
        best = int(row.min())
        # second best: take smallest two
        sorted_vals = np.partition(row, 1)[:2]
        second = int(sorted_vals[1]) if len(sorted_vals) > 1 else 10**9
        gap = second - best

        chosen_c = row_to_col[r]
        chosen = int(cost[r, chosen_c])

        row_ok = (gap >= gap_threshold) and (chosen <= max_dist)
        if not row_ok:
            confident = False

        per_row.append({
            "row": r,
            "best": best,
            "second": second,
            "gap": gap,
            "chosen": chosen,
            "ok": row_ok
        })

    total = int(sum(cost[r, c] for r, c in zip(row_ind, col_ind)))
    avg = float(total) / float(n)

    report = {
        "total_cost": total,
        "avg_cost": avg,
        "gap_threshold": gap_threshold,
        "max_dist": max_dist,
        "per_row": per_row,
    }
    return confident, report

############     ################
############ OCR ################
############     ################


# -----------------------------
# Text normalization
# -----------------------------

def normalize_text(s: str) -> str:
    """
    Normalize in a way that tends to help comparisons:
    - Unicode normalize
    - lowercase
    - collapse whitespace
    - remove “weird” punctuation except basic ones (tweak to your needs)
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()

    # keep letters/numbers/basic punctuation; drop the rest
    s = re.sub(r"[^\w\s\.\,\:\;\-\(\)\/%]", " ", s, flags=re.UNICODE)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Similarity measures
# -----------------------------

def sequence_similarity(a: str, b: str) -> float:
    """
    Character-level similarity in [0,1].
    """
    return SequenceMatcher(None, a, b).ratio()


def jaccard_similarity_tokens(a: str, b: str) -> float:
    """
    Token-level Jaccard similarity in [0,1].
    """
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def containment_similarity(a: str, b: str) -> float:
    """
    How much of the shorter text's tokens are contained in the longer text.
    Useful when one side has extra boilerplate.
    """
    ta = a.split()
    tb = b.split()
    sa, sb = set(ta), set(tb)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    small, big = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return len(small & big) / max(1, len(small))


def compare_pages_same_section(raw1, raw2):
    n1 = normalize_text(raw1)
    n2 = normalize_text(raw2)

    return {
        "raw_text_1": raw1,
        "raw_text_2": raw2,
        "normalized_text_1": n1,
        "normalized_text_2": n2,
        "similarity_sequence": sequence_similarity(n1, n2),
        "similarity_jaccard_tokens": jaccard_similarity_tokens(n1, n2),
        "similarity_containment": containment_similarity(n1, n2),
    }

def match_pages_text(pages_list,templates_to_consider,similarity):
    #assume phash are already computed
    cost = similarity*(-1)+1
    # Hungarian assignment (minimize total cost)
    assignement = linear_sum_assignment(cost)
    row_ind, col_ind = assignement

    '''if the dimension of the two lists is different -> Every single template will be assigned to a page.
    The function will pick the subset of pages that results in the lowest total Hamming distance.
    The "leftover" pages will be ignored. Because linear_sum_assignment only returns a match for the smaller dimension, 
    the row_ind and col_ind arrays will only have a length equal to the number of templates'''
    
    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append({
            "page_index": pages_list[i],
            "template_index": templates_to_consider[j],
            "similarity": int(cost[i, j]),
        })
    
    # confident,report = assignment_confidence_text(cost, assignement, gap_threshold=gap_threshold, max_dist=max_dist)
    # decide how to evaluate the confidence

    matches_sorted = sorted(matches, key=lambda x: x["page_index"])

    return matches_sorted, cost #, confident, report

def assignment_confidence_text(cost: np.ndarray, assignment, gap_threshold: int, max_dist: int):
    """
    Returns (is_confident, report_dict).
    Confidence logic:
      - per-row gap test: (2nd best - best) >= gap_threshold
      - matched distance <= max_dist
    """
    n = cost.shape[0]
    row_to_col = {r: c for r, c in assignment}

    per_row = []
    confident = True

    for r in range(n):
        row = cost[r].astype(int)
        best = int(row.min())
        # second best: take smallest two
        sorted_vals = np.partition(row, 1)[:2]
        second = int(sorted_vals[1]) if len(sorted_vals) > 1 else 10**9
        gap = second - best

        chosen_c = row_to_col[r]
        chosen = int(cost[r, chosen_c])

        row_ok = (gap >= gap_threshold) and (chosen <= max_dist)
        if not row_ok:
            confident = False

        per_row.append({
            "row": r,
            "best": best,
            "second": second,
            "gap": gap,
            "chosen": chosen,
            "ok": row_ok
        })

    total = int(sum(cost[r, c] for r, c in assignment))
    avg = float(total) / float(n)

    report = {
        "total_cost": total,
        "avg_cost": avg,
        "gap_threshold": gap_threshold,
        "max_dist": max_dist,
        "per_row": per_row,
    }
    return confident, report