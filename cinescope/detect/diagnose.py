"""Where does a face detector lose AP? Break results down by face size.

For every ground-truth face: the best IoU any detection reaches with it, and
the score of that detection. Aggregated per size bucket, this separates the
three failure modes that AP alone mixes together:

  missed        no detection overlaps the face at all (best IoU < 0.3)
  mislocalised  a detection is there but too loose/tight (0.3 <= IoU < 0.5)
  low score     found (IoU >= 0.5) but ranked below false positives

plus duplicates: extra detections with IoU >= 0.5 on a face already found,
which count as false positives.
"""
from __future__ import annotations

import numpy as np

from .boxes import iou, xywh_to_xyxy

BUCKETS = ((0, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 1e9))


def face_size(xywh: np.ndarray) -> np.ndarray:
    """sqrt(w*h): one number per face, robust to aspect ratio."""
    b = np.asarray(xywh, np.float64).reshape(-1, 4)
    return np.sqrt(np.clip(b[:, 2], 0, None) * np.clip(b[:, 3], 0, None))


def per_face(gt_xywh: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    """gt (M,4) xywh; pred (N,5) xywh+score. Per face: best IoU, score of that
    detection, and per image the number of duplicate detections per face."""
    g = xywh_to_xyxy(gt_xywh)
    p = np.asarray(pred, np.float64).reshape(-1, 5)
    M = len(g)
    if not M:
        return {"best_iou": np.zeros(0), "score": np.zeros(0), "dups": np.zeros(0, int)}
    if not len(p):
        return {"best_iou": np.zeros(M), "score": np.zeros(M), "dups": np.zeros(M, int)}
    ov = iou(g, xywh_to_xyxy(p[:, :4]))                    # (M, N)
    j = ov.argmax(1)
    hits = ov >= 0.5
    # a detection counts for the face it overlaps most; more than one per face = duplicates
    owner = ov.argmax(0)
    counts = np.array([(hits[m] & (owner == m)).sum() for m in range(M)])
    return {"best_iou": ov[np.arange(M), j], "score": p[j, 4], "dups": np.clip(counts - 1, 0, None)}


def breakdown(gts: dict[str, np.ndarray], preds: dict[str, np.ndarray],
              keep: dict[str, np.ndarray] | None = None, score_thresh: float = 0.5) -> list[dict]:
    """Rows per size bucket. keep[image] = indices of faces to include (e.g. the
    official Easy list); None = every face."""
    sizes, best, score, dups = [], [], [], []
    for k, g in gts.items():
        g = np.asarray(g, np.float64).reshape(-1, 4)
        if keep is not None:
            g = g[np.asarray(keep.get(k, []), int)]
        if not len(g):
            continue
        r = per_face(g, preds.get(k, np.zeros((0, 5))))
        sizes.append(face_size(g))
        best.append(r["best_iou"])
        score.append(r["score"])
        dups.append(r["dups"])
    if not sizes:
        return []
    sizes, best, score, dups = map(np.concatenate, (sizes, best, score, dups))
    rows = []
    for lo, hi in BUCKETS:
        m = (sizes >= lo) & (sizes < hi)
        n = int(m.sum())
        if not n:
            continue
        found = best[m] >= 0.5
        rows.append({
            "size": f"{int(lo)}-{int(hi) if hi < 1e8 else '∞'} px", "faces": n,
            "found_any_score": round(float(found.mean()), 3),
            f"found_score>={score_thresh}": round(float((found & (score[m] >= score_thresh)).mean()), 3),
            "mislocalised": round(float(((best[m] >= 0.3) & ~found).mean()), 3),
            "missed": round(float((best[m] < 0.3).mean()), 3),
            "median_iou_found": round(float(np.median(best[m][found])), 3) if found.any() else None,
            "median_score_found": round(float(np.median(score[m][found])), 3) if found.any() else None,
            "duplicates_per_face": round(float(dups[m].mean()), 3),
        })
    return rows


def markdown(rows: list[dict]) -> str:
    if not rows:
        return "(no faces)"
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join("—" if r[c] is None else str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)
