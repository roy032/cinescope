"""Multi-object tracking metrics from scratch: CLEAR MOT (MOTA) and IDF1.

MOTA (Bernardin & Stiefelhagen, 2008) counts per-frame mistakes:

    MOTA = 1 - (FN + FP + IDSW) / number of ground-truth boxes

matching ground truth (GT) to hypotheses (tracker output) frame by frame. A
correspondence from the previous frame is kept while it still overlaps
(IoU >= threshold) -- the tracker should not be punished for a better-looking
but different assignment -- and the rest are matched with the Hungarian
algorithm. An identity switch is a GT object matched to a different hypothesis
id than the last time it was matched. MOTA is dominated by detection errors
(FN, FP): one ID switch costs as much as one missed box.

IDF1 (Ristani et al., 2016) asks the identity question directly: find the
single best one-to-one mapping between *whole* GT trajectories and *whole*
hypothesis trajectories (bipartite matching on how many frames each pair
overlaps), then

    IDF1 = 2 IDTP / (number of GT boxes + number of hypothesis boxes).

A tracker that follows a face perfectly but swaps its id halfway through
gets MOTA ~1 (one switch) but IDF1 ~0.5 -- which is why a character timeline
cares about IDF1 more.

Frames: dict frame -> (ids, boxes xyxy). GT frames may also carry an `ignore`
list of boxes (unidentifiable faces): a hypothesis on one is neither TP nor FP.
Only frames present in `gt` are scored, so sparsely labelled clips (every
12th frame, say) work: the hypotheses on other frames are skipped.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..detect.boxes import iou
from .hungarian import linear_sum_assignment

Frames = dict[int, tuple[list, np.ndarray]]


def _filter_ignored(hb: np.ndarray, hid: list, ign: np.ndarray, matched: set[int], thr: float):
    if ign is None or not len(ign) or not len(hb):
        return set()
    ov = iou(hb, ign).max(1)
    return {j for j in range(len(hb)) if j not in matched and ov[j] >= thr}


def evaluate(gt: Frames, hyp: Frames, iou_thr: float = 0.5,
             ignore: dict[int, np.ndarray] | None = None) -> dict:
    ignore = ignore or {}
    fn = fp = idsw = tp = 0
    n_gt = 0
    iou_sum = 0.0
    last: dict = {}                       # gt id -> hyp id it was last matched to
    prev_pairs: dict = {}                 # gt id -> hyp id matched in the previous scored frame
    overlap = defaultdict(int)            # (gt id, hyp id) -> frames with IoU >= thr (for IDF1)
    gt_count = defaultdict(int)
    hyp_count = defaultdict(int)
    mostly = defaultdict(lambda: [0, 0])  # gt id -> [frames matched, frames present]

    for f in sorted(gt):
        gid, gb = gt[f]
        gid = list(gid)
        gb = np.asarray(gb, dtype=np.float64).reshape(-1, 4)
        hid, hb = hyp.get(f, ([], np.zeros((0, 4))))
        hid = list(hid)
        hb = np.asarray(hb, dtype=np.float64).reshape(-1, 4)
        ious = iou(gb, hb) if len(gb) and len(hb) else np.zeros((len(gb), len(hb)))

        # Per-frame CLEAR matching.
        pairs: dict[int, int] = {}
        hpos = {h: j for j, h in enumerate(hid)}
        for i, g in enumerate(gid):                         # keep last frame's correspondences
            h = prev_pairs.get(g)
            if h is not None and h in hpos and ious[i, hpos[h]] >= iou_thr and hpos[h] not in pairs.values():
                pairs[i] = hpos[h]
        free_g = [i for i in range(len(gid)) if i not in pairs]
        free_h = [j for j in range(len(hid)) if j not in pairs.values()]
        if free_g and free_h:
            sub = ious[np.ix_(free_g, free_h)]
            cost = np.where(sub >= iou_thr, 1.0 - sub, 1e6)
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c, strict=True):
                if sub[a, b] >= iou_thr:
                    pairs[free_g[a]] = free_h[b]
        ign = _filter_ignored(hb, hid, ignore.get(f), set(pairs.values()), iou_thr)

        new_prev = {}
        for i, j in pairs.items():
            g, h = gid[i], hid[j]
            if g in last and last[g] != h:
                idsw += 1
            last[g] = h
            new_prev[g] = h
            iou_sum += ious[i, j]
            mostly[g][0] += 1
        prev_pairs = new_prev
        tp += len(pairs)
        fn += len(gid) - len(pairs)
        fp += len(hid) - len(pairs) - len(ign)
        n_gt += len(gid)
        for i, g in enumerate(gid):
            gt_count[g] += 1
            mostly[g][1] += 1
            for j, h in enumerate(hid):
                if ious[i, j] >= iou_thr:
                    overlap[(g, h)] += 1
        for j, h in enumerate(hid):
            if j not in ign:
                hyp_count[h] += 1

    # IDF1: maximise the number of frames shared by matched trajectory pairs.
    gids, hids = sorted(gt_count, key=str), sorted(hyp_count, key=str)
    idtp = 0
    if gids and hids:
        w = np.zeros((len(gids), len(hids)))
        gi = {g: i for i, g in enumerate(gids)}
        hi = {h: j for j, h in enumerate(hids)}
        for (g, h), k in overlap.items():
            if h in hi:
                w[gi[g], hi[h]] = k
        r, c = linear_sum_assignment(w, maximize=True)
        idtp = int(w[r, c].sum())
    n_hyp = sum(hyp_count.values())
    n_gt_ids = sum(gt_count.values())
    ratio = [m / max(n, 1) for m, n in mostly.values()]
    return {
        "mota": 1.0 - (fn + fp + idsw) / max(n_gt, 1),
        "motp_iou": iou_sum / max(tp, 1),
        "idf1": 2 * idtp / max(n_gt_ids + n_hyp, 1),
        "idp": idtp / max(n_hyp, 1),
        "idr": idtp / max(n_gt_ids, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(n_gt, 1),
        "tp": tp, "fp": fp, "fn": fn, "idsw": idsw, "num_gt": n_gt, "num_hyp": n_hyp,
        "gt_ids": len(gt_count), "hyp_ids": len(hyp_count),
        "mostly_tracked": int(sum(r >= 0.8 for r in ratio)),
        "mostly_lost": int(sum(r < 0.2 for r in ratio)),
    }


def tracks_to_frames(tracks, frames: set[int] | None = None) -> Frames:
    """[Track] (or dicts with id/frames/boxes) -> {frame: (ids, boxes)}."""
    out: dict[int, tuple[list, list]] = defaultdict(lambda: ([], []))
    for t in tracks:
        tid = t.id if hasattr(t, "id") else t["id"]
        fr = t.frames if hasattr(t, "frames") else t["frames"]
        bx = t.boxes if hasattr(t, "boxes") else t["boxes"]
        for f, b in zip(fr, bx, strict=True):
            if frames is None or f in frames:
                out[f][0].append(tid)
                out[f][1].append(b)
    return {f: (ids, np.asarray(b, dtype=np.float64).reshape(-1, 4)) for f, (ids, b) in out.items()}
