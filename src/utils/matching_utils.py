import cv2
import numpy as np
from scipy.fftpack import dct
from scipy.optimize import linear_sum_assignment

'''cost_matrix, matches = match_pages(folder, folder_2, border_crop_pct=crop_pct)
assignment = hungarian_min_cost(cost_matrix)

confident, report = assignment_confidence_phash(
    cost_matrix, assignment, gap_threshold=5, max_dist=18
)

print(f"Border crop: {crop_pct*100:.1f}% each side")
print("\n=== pHash cost matrix (Hamming distances) ===")
print(cost_matrix)
print("\n=== pHash assignment (Hungarian) ===")
print("\nMatches (A -> B):")
for m in matches:
    print(f"A[{m['A_index']}] {m['A_file']}  ->  B[{m['B_index']}] {m['B_file']}   (dist={m['hamming']})")

print("\n=== pHash confidence report ===")
print(f"Total cost: {report['total_cost']}, Avg: {report['avg_cost']:.2f}")
for r in report["per_row"]:
    print(f"Row A[{r['row']}]: best={r['best']} second={r['second']} gap={r['gap']} chosen={r['chosen']} ok={r['ok']}")
print(f"\nOverall confident: {confident}")'''


def match_pages_phash(page_dict, template_dict, pages_list, templates_to_consider, border_crop_pct: float = 0.0):
    #assume phash are already computed

    hashes_pages = []
    hashes_templates = []

    #length of pages and templates may be different if templates only has the ones to censor
    for img_id in pages_list:
        hashes_pages.append(page_dict[img_id]['page_phash'])
    for t_id in templates_to_consider:
        hashes_templates.append(template_dict[t_id]['page_phash'])

    # Cost matrix: Hamming distances
    cost = np.zeros((len(paths_a), len(paths_b)), dtype=np.int32)
    for i in range(len(paths_a)):
        for j in range(len(paths_b)):
            cost[i, j] = hamming_distance(hashes_a[i], hashes_b[j])

    # Hungarian assignment (minimize total cost)
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append({
            "A_index": i,
            "A_file": os.path.basename(paths_a[i]),
            "B_index": j,
            "B_file": os.path.basename(paths_b[j]),
            "hamming": int(cost[i, j]),
        })

    matches_sorted = sorted(matches, key=lambda x: x["A_index"])
    return cost, matches_sorted


def hungarian_min_cost(cost: np.ndarray):
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def phash_cost_matrix(pathsA, pathsB, border_crop_pct: float):
    hashesA, hashesB = [], []
    for p in pathsA:
        with Image.open(p) as im:
            hashesA.append(phash(im, border_crop_pct=border_crop_pct))
    for p in pathsB:
        with Image.open(p) as im:
            hashesB.append(phash(im, border_crop_pct=border_crop_pct))

    cost = np.zeros((len(pathsA), len(pathsB)), dtype=np.int32)
    for i in range(len(pathsA)):
        for j in range(len(pathsB)):
            cost[i, j] = hamming_distance(hashesA[i], hashesB[j])
    return cost


def assignment_confidence_phash(cost: np.ndarray, assignment, gap_threshold: int, max_dist: int):
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