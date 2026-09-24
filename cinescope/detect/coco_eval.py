"""COCO-style detection evaluation, re-implemented from the protocol.

Follows pycocotools' COCOeval (bbox) step by step, so the numbers can be
checked against it (tests/test_week6.py does, when pycocotools is installed):

1. Per image and category, detections are sorted by score and cut to the
   largest maxDets (100). Ground truths outside the area range, or marked
   ignore, are "ignored": matching one neither helps nor hurts.
2. Greedy matching per IoU threshold t ∈ {0.50, 0.55, …, 0.95}: each detection,
   in score order, takes the unmatched ground truth with the highest IoU ≥ t,
   preferring non-ignored ones. Crowd ground truths can absorb any number of
   detections. An unmatched detection outside the area range is ignored too.
3. Per category, all images' detections are merged and sorted by score.
   Cumulative TP/FP give precision and recall at every rank; precision is
   made monotonically non-increasing from the right (the "interpolation"),
   then sampled at 101 recall points 0, 0.01, …, 1.
4. AP = mean of those 101 values; mAP@[.5:.95] = mean over categories and the
   ten IoU thresholds. Categories without ground truth are left out.

Why 0.5:0.95: AP@0.5 rewards a box that is roughly right; averaging up to 0.95
rewards boxes that are *tight*, which is what localisation quality means.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .boxes import iou, xywh_to_xyxy

AREA_RANGES = {"all": (0.0, 1e10), "small": (0.0, 32.0 ** 2), "medium": (32.0 ** 2, 96.0 ** 2),
               "large": (96.0 ** 2, 1e10)}


@dataclass
class EvalParams:
    iou_thrs: np.ndarray = field(default_factory=lambda: np.linspace(0.5, 0.95, 10))
    rec_thrs: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 1.0, 101))
    max_dets: tuple[int, ...] = (1, 10, 100)
    area_ranges: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(AREA_RANGES))


def _evaluate_image(gts: list[dict], dts: list[dict], a_rng: tuple[float, float], max_det: int,
                    iou_thrs: np.ndarray) -> dict | None:
    if not gts and not dts:
        return None
    g_ignore = np.array([bool(g.get("ignore", 0)) or bool(g.get("iscrowd", 0))
                         or not (a_rng[0] <= g["area"] <= a_rng[1]) for g in gts], dtype=bool)
    # the ignore flag COCOeval uses: explicit ignore, crowd, or out of area range
    g_order = np.argsort(g_ignore, kind="stable")               # non-ignored first
    gts = [gts[i] for i in g_order]
    g_ignore = g_ignore[g_order]
    d_order = np.argsort([-d["score"] for d in dts], kind="stable")[:max_det]
    dts = [dts[i] for i in d_order]
    crowd = np.array([bool(g.get("iscrowd", 0)) for g in gts], dtype=bool)

    T, G, D = len(iou_thrs), len(gts), len(dts)
    gtm = np.full((T, G), -1, dtype=np.int64)
    dtm = np.full((T, D), -1, dtype=np.int64)
    dt_ig = np.zeros((T, D), dtype=bool)
    if G and D:
        ious = iou(xywh_to_xyxy([d["bbox"] for d in dts]), xywh_to_xyxy([g["bbox"] for g in gts]), crowd)
        for ti, t in enumerate(iou_thrs):
            for di in range(D):
                best, m = min(t, 1 - 1e-10), -1
                for gi in range(G):
                    if gtm[ti, gi] >= 0 and not crowd[gi]:      # taken, and not a crowd
                        continue
                    if m > -1 and not g_ignore[m] and g_ignore[gi]:
                        break                                   # already on a real gt; rest are ignored
                    if ious[di, gi] < best:
                        continue
                    best, m = ious[di, gi], gi
                if m == -1:
                    continue
                dt_ig[ti, di] = g_ignore[m]
                dtm[ti, di] = m
                gtm[ti, m] = di
    d_area = np.array([d["bbox"][2] * d["bbox"][3] for d in dts], dtype=np.float64)
    out_of_range = (d_area < a_rng[0]) | (d_area > a_rng[1])
    dt_ig |= (dtm < 0) & out_of_range[None, :]
    return {"dt_scores": np.array([d["score"] for d in dts], dtype=np.float64),
            "dt_matched": dtm >= 0, "dt_ignore": dt_ig, "gt_ignore": g_ignore}


def _precision_at_recalls(tp: np.ndarray, fp: np.ndarray, n_pos: int, rec_thrs: np.ndarray
                          ) -> tuple[np.ndarray, float]:
    tp_c, fp_c = np.cumsum(tp, dtype=np.float64), np.cumsum(fp, dtype=np.float64)
    rc = tp_c / n_pos
    pr = tp_c / (tp_c + fp_c + np.spacing(1))
    recall = float(rc[-1]) if len(rc) else 0.0
    pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr    # monotone envelope
    q = np.zeros(len(rec_thrs))
    inds = np.searchsorted(rc, rec_thrs, side="left")
    valid = inds < len(pr)
    q[valid] = pr[inds[valid]]
    return q, recall


class COCOEvaluator:
    """Evaluate detections against COCO-format ground truth.

        ev = COCOEvaluator(gt_dict)              # {"images", "annotations", "categories"}
        stats = ev.evaluate(detections)          # [{"image_id", "category_id", "bbox" (xywh), "score"}]
        stats["AP"], stats["AP50"], stats["AP_small"], ...

    Annotation "area" defaults to w*h when absent (COCO stores mask area there).
    One deliberate difference: an explicit `"ignore": 1` on a ground truth is
    honoured (pycocotools overwrites it with `iscrowd`); WIDER FACE's
    too-small/invalid faces need it. Without such flags the numbers match.
    """

    def __init__(self, gt: dict, params: EvalParams | None = None):
        self.p = params or EvalParams()
        self.img_ids = sorted(im["id"] for im in gt["images"])
        self.cat_ids = sorted(c["id"] for c in gt["categories"])
        self.gts: dict[tuple, list[dict]] = defaultdict(list)
        for a in gt["annotations"]:
            a = dict(a)
            a.setdefault("area", a["bbox"][2] * a["bbox"][3])
            self.gts[a["image_id"], a["category_id"]].append(a)

    def evaluate(self, dets: list[dict]) -> dict:
        p = self.p
        dts: dict[tuple, list[dict]] = defaultdict(list)
        for d in dets:
            dts[d["image_id"], d["category_id"]].append(d)
        T, R, K = len(p.iou_thrs), len(p.rec_thrs), len(self.cat_ids)
        A, M = len(p.area_ranges), len(p.max_dets)
        precision = -np.ones((T, R, K, A, M))
        recall = -np.ones((T, K, A, M))
        max_det_all = max(p.max_dets)
        for ki, c in enumerate(self.cat_ids):
            for ai, a_rng in enumerate(p.area_ranges.values()):
                evals = [e for e in (_evaluate_image(self.gts.get((i, c), []), dts.get((i, c), []),
                                                     a_rng, max_det_all, p.iou_thrs)
                                     for i in self.img_ids) if e is not None]
                if not evals:
                    continue
                n_pos = int(sum((~e["gt_ignore"]).sum() for e in evals))
                if n_pos == 0:
                    continue
                for mi, md in enumerate(p.max_dets):
                    scores = np.concatenate([e["dt_scores"][:md] for e in evals])
                    order = np.argsort(-scores, kind="mergesort")
                    matched = np.concatenate([e["dt_matched"][:, :md] for e in evals], axis=1)[:, order]
                    ignored = np.concatenate([e["dt_ignore"][:, :md] for e in evals], axis=1)[:, order]
                    tps = matched & ~ignored
                    fps = ~matched & ~ignored
                    for ti in range(T):
                        q, r = _precision_at_recalls(tps[ti], fps[ti], n_pos, p.rec_thrs)
                        precision[ti, :, ki, ai, mi] = q
                        recall[ti, ki, ai, mi] = r
        self.precision, self.recall = precision, recall
        return self.summarize()

    def _mean(self, arr: np.ndarray) -> float:
        v = arr[arr > -1]
        return float(v.mean()) if v.size else -1.0

    def summarize(self) -> dict:
        p, P, Rc = self.p, self.precision, self.recall
        areas = list(p.area_ranges)
        a = {k: i for i, k in enumerate(areas)}
        m = {d: i for i, d in enumerate(p.max_dets)}
        top = max(p.max_dets)
        t50 = int(np.argmin(np.abs(p.iou_thrs - 0.5)))
        t75 = int(np.argmin(np.abs(p.iou_thrs - 0.75)))
        out = {
            "AP": self._mean(P[:, :, :, a["all"], m[top]]),
            "AP50": self._mean(P[t50, :, :, a["all"], m[top]]),
            "AP75": self._mean(P[t75, :, :, a["all"], m[top]]),
        }
        for name in ("small", "medium", "large"):
            if name in a:
                out[f"AP_{name}"] = self._mean(P[:, :, :, a[name], m[top]])
        for d in p.max_dets:
            out[f"AR{d}"] = self._mean(Rc[:, :, a["all"], m[d]])
        for name in ("small", "medium", "large"):
            if name in a:
                out[f"AR_{name}"] = self._mean(Rc[:, :, a[name], m[top]])
        out["per_category_AP"] = {c: self._mean(P[:, :, k, a["all"], m[top]])
                                  for k, c in enumerate(self.cat_ids)}
        return out


def ap_at_iou(gt_boxes: list[np.ndarray], det_boxes: list[np.ndarray], det_scores: list[np.ndarray],
              iou_thresh: float = 0.5, gt_ignore: list[np.ndarray] | None = None) -> dict:
    """Single-class AP at one IoU threshold with all-point interpolation (the
    VOC 2010+ / WIDER FACE style: area under the monotone PR envelope, not the
    101-point sample). Boxes are xyxy; one list entry per image. Detections
    matched to ignored ground truth (WIDER FACE's tiny/invalid faces) are
    neither TP nor FP."""
    rows = []
    n_pos = 0
    for i, (g, d, s) in enumerate(zip(gt_boxes, det_boxes, det_scores, strict=True)):
        g = np.asarray(g, dtype=np.float64).reshape(-1, 4)
        d = np.asarray(d, dtype=np.float64).reshape(-1, 4)
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        ign = np.zeros(len(g), bool) if gt_ignore is None else np.asarray(gt_ignore[i], bool)
        n_pos += int((~ign).sum())
        order = np.argsort(-s, kind="stable")
        d, s = d[order], s[order]
        taken = np.zeros(len(g), bool)
        ov = iou(d, g) if len(g) and len(d) else np.zeros((len(d), len(g)))
        for j in range(len(d)):
            if len(g):
                cand = np.where(taken, -1.0, ov[j])
                k = int(np.argmax(cand))
                if cand[k] >= iou_thresh:
                    taken[k] = True
                    rows.append((s[j], 0 if ign[k] else 1, 0, bool(ign[k])))
                    continue
            rows.append((s[j], 0, 1, False))
    if n_pos == 0:
        return {"ap": float("nan"), "n_pos": 0, "precision": np.zeros(0), "recall": np.zeros(0)}
    rows = [r for r in rows if not r[3]]
    rows.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in rows], dtype=np.float64)
    fp = np.cumsum([r[2] for r in rows], dtype=np.float64)
    rec = tp / n_pos
    prec = tp / np.maximum(tp + fp, np.finfo(float).eps)
    if not len(rows):
        return {"ap": 0.0, "n_pos": n_pos, "precision": prec, "recall": rec}
    mrec = np.r_[0.0, rec, 1.0]
    mpre = np.maximum.accumulate(np.r_[0.0, prec, 0.0][::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {"ap": ap, "n_pos": n_pos, "precision": prec, "recall": rec}
