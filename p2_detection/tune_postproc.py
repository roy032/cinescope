"""Can post-processing fix the duplicate boxes on large faces? (no retraining)

diagnose_wider.py showed that faces over 256 px attract 0.3 extra boxes each and
that Easy, the set made of large faces, scores lowest. A duplicate is a false
positive at a high score, which is exactly what AP punishes. This script replays
the saved validation predictions (eval_wider.py's wider_val_preds.npz, already
after NMS at IoU 0.5) through stricter suppression rules and re-scores each with
the official WIDER protocol. It needs only the ground-truth .mat files, not the
images or a GPU.

    python p2_detection/tune_postproc.py --preds p2_detection/results/centernet_r18_log/wider_val_preds.npz \
        --eval-tools path/to/eval_tools/ground_truth

The rules compared:
  nms@t        plain NMS at a lower IoU
  contain@t    also drop a box when t of the SMALLER box lies inside a higher-scoring one
               (two peaks on one face give a tight box inside a loose one: low IoU, high overlap)
  large-nms@t  NMS at IoU t only between boxes of at least 64 px (crowds of small faces untouched)

Caveat: rules are chosen and reported on the same validation set (WIDER test
labels are not public), so a small gain is an optimistic estimate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.detect.boxes import intersection, xywh_to_xyxy  # noqa: E402
from cinescope.detect.nms import nms  # noqa: E402

HERE = Path(__file__).resolve().parent


def load_preds(path: str | Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {str(k): z[f"p{i}"].astype(np.float64) for i, k in enumerate(z["keys"])}


def suppress_contained(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> np.ndarray:
    """Greedy, like NMS, but the overlap is intersection / area of the smaller box."""
    order = np.argsort(-scores, kind="stable")
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if not rest.size:
            break
        inter = intersection(boxes[i:i + 1], boxes[rest])[0]
        overlap = inter / np.maximum(np.minimum(areas[i], areas[rest]), 1e-9)
        order = rest[overlap <= thresh]
    return np.asarray(keep, dtype=np.int64)


def large_nms(boxes: np.ndarray, scores: np.ndarray, thresh: float, min_side: float = 64.0) -> np.ndarray:
    """NMS at `thresh` among boxes whose shorter side is >= min_side; the rest are kept."""
    side = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    big = np.flatnonzero(side >= min_side)
    small = np.flatnonzero(side < min_side)
    kept_big = big[nms(boxes[big], scores[big], thresh)] if big.size else big
    return np.sort(np.concatenate([small, kept_big]))


def apply(preds: dict[str, np.ndarray], rule: str, t: float) -> dict[str, np.ndarray]:
    out = {}
    for k, p in preds.items():
        if len(p) < 2 or rule == "base":
            out[k] = p
            continue
        b, s = xywh_to_xyxy(p[:, :4]), p[:, 4]
        if rule == "nms":
            keep = nms(b, s, t)
        elif rule == "contain":
            keep = suppress_contained(b, s, t)
        elif rule == "large-nms":
            keep = large_nms(b, s, t)
        else:
            raise ValueError(rule)
        out[k] = p[np.sort(keep)]
    return out


RULES = [("base", 0.0), ("nms", 0.4), ("nms", 0.3), ("contain", 0.9), ("contain", 0.8), ("contain", 0.7),
         ("large-nms", 0.3), ("large-nms", 0.2)]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--eval-tools", required=True, help="folder with wider_face_val.mat and wider_{easy,medium,hard}_val.mat")
    ap.add_argument("--out", help="write a markdown table here")
    args = ap.parse_args(argv)

    from cinescope.detect.wider_eval import evaluate, load_ground_truth

    gts, keeps = load_ground_truth(args.eval_tools)
    preds = load_preds(args.preds)
    rows = []
    for rule, t in RULES:
        p = apply(preds, rule, t)
        ap_ = evaluate(p, gts, keeps)
        n = sum(len(v) for v in p.values())
        rows.append({"rule": rule if rule == "base" else f"{rule}@{t}", "boxes": n, **{s: round(v, 4) for s, v in ap_.items()}})
        print(json.dumps(rows[-1]), flush=True)
    base = rows[0]
    for r in rows:
        r["mean"] = round(float(np.mean([r["easy"], r["medium"], r["hard"]])), 4)
    best = max(rows, key=lambda r: r["mean"])
    lines = [f"Post-processing on `{Path(args.preds).parent.name}` (official WIDER val AP; base = NMS 0.5):", "",
             "| rule | boxes kept | Easy | Medium | Hard | mean |", "|---|---|---|---|---|---|"]
    for r in rows:
        mark = " **(best)**" if r is best and r is not base else ""
        lines.append(f"| {r['rule']}{mark} | {r['boxes']} | {r['easy']:.3f} | {r['medium']:.3f} | {r['hard']:.3f} | {r['mean']:.3f} |")
    md = "\n".join(lines)
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n", encoding="utf-8")
    return {"rows": rows, "best": best}


if __name__ == "__main__":
    main()
