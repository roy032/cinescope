"""WIDER FACE evaluation, as the official toolkit computes it.

The benchmark's Easy / Medium / Hard numbers come from its MATLAB toolkit
(eval_tools/, with wider_face_val.mat and wider_{easy,medium,hard}_val.mat).
This is a port of that procedure, so numbers are comparable with papers:

  * every prediction score is min-max normalised over the whole val set;
  * per image, predictions (sorted by score) are matched to ground truth at
    IoU >= 0.5 using the "+1" pixel convention of the original code; faces not
    in the setting's keep-list are *ignored*: a prediction on one is removed
    from the proposal count instead of becoming a false positive;
  * precision/recall are accumulated at 1,000 score thresholds over the whole
    set and AP is the VOC all-point area under that curve.

The three settings differ only in which faces must be found: Easy keeps the
large, clear faces; Hard keeps all of them down to ~10 px, so every detector
scores lowest there.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SETTINGS = ("easy", "medium", "hard")


def _overlaps_plus1(boxes: np.ndarray, query: np.ndarray) -> np.ndarray:
    """IoU with the original code's inclusive-pixel (+1) widths; xyxy inputs."""
    bw = boxes[:, 2] - boxes[:, 0] + 1
    bh = boxes[:, 3] - boxes[:, 1] + 1
    qw = query[:, 2] - query[:, 0] + 1
    qh = query[:, 3] - query[:, 1] + 1
    iw = np.minimum(boxes[:, None, 2], query[None, :, 2]) - np.maximum(boxes[:, None, 0], query[None, :, 0]) + 1
    ih = np.minimum(boxes[:, None, 3], query[None, :, 3]) - np.maximum(boxes[:, None, 1], query[None, :, 1]) + 1
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    union = (bw * bh)[:, None] + (qw * qh)[None, :] - inter
    return np.where((iw > 0) & (ih > 0), inter / union, 0.0)


def image_eval(pred: np.ndarray, gt: np.ndarray, keep: np.ndarray, iou_thresh: float = 0.5
               ) -> tuple[np.ndarray, np.ndarray]:
    """pred (N,5) xywh+score sorted by score desc; gt (M,4) xywh; keep (M,) bool.
    Returns (faces recalled after each prediction, proposal flags: 1 counts, -1 ignored)."""
    p = pred[:, :4].astype(np.float64).copy()
    g = gt.astype(np.float64).copy()
    p[:, 2:] += p[:, :2]
    g[:, 2:] += g[:, :2]
    overlaps = _overlaps_plus1(p, g)
    recall_list = np.zeros(len(g))
    proposal = np.ones(len(p))
    pred_recall = np.zeros(len(p))
    for h in range(len(p)):
        k = int(overlaps[h].argmax())
        if overlaps[h, k] >= iou_thresh:
            if not keep[k]:
                recall_list[k] = -1
                proposal[h] = -1
            elif recall_list[k] == 0:
                recall_list[k] = 1
        pred_recall[h] = (recall_list == 1).sum()
    return pred_recall, proposal


def _img_pr_info(n_thresh: int, pred: np.ndarray, proposal: np.ndarray, pred_recall: np.ndarray) -> np.ndarray:
    info = np.zeros((n_thresh, 2))
    for t in range(n_thresh):
        thresh = 1 - (t + 1) / n_thresh
        idx = np.where(pred[:, 4] >= thresh)[0]
        if len(idx):
            last = idx[-1]
            info[t, 0] = (proposal[:last + 1] == 1).sum()
            info[t, 1] = pred_recall[last]
    return info


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.r_[0.0, rec, 1.0]
    mpre = np.r_[0.0, prec, 0.0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def evaluate(preds: dict[str, np.ndarray], gts: dict[str, np.ndarray], keeps: dict[str, dict[str, np.ndarray]],
             iou_thresh: float = 0.5, n_thresh: int = 1000) -> dict[str, float]:
    """preds[image] (N,5) xywh+score; gts[image] (M,4) xywh; keeps[setting][image]
    = 0-based indices of the faces that count in that setting."""
    scores = np.concatenate([p[:, 4] for p in preds.values() if len(p)] or [np.zeros(0)])
    lo, hi = (scores.min(), scores.max()) if len(scores) else (0.0, 1.0)
    span = hi - lo if hi > lo else 1.0
    normed = {}
    for k, p in preds.items():
        p = np.asarray(p, dtype=np.float64).reshape(-1, 5).copy()
        p[:, 4] = (p[:, 4] - lo) / span
        normed[k] = p[np.argsort(-p[:, 4], kind="stable")]
    out = {}
    for setting, keep_map in keeps.items():
        pr = np.zeros((n_thresh, 2))
        n_faces = 0
        for img, gt in gts.items():
            keep_idx = np.asarray(keep_map.get(img, []), dtype=np.int64)
            n_faces += len(keep_idx)
            p = normed.get(img, np.zeros((0, 5)))
            if not len(gt) or not len(p):
                continue
            keep = np.zeros(len(gt), bool)
            keep[keep_idx] = True
            rec, prop = image_eval(p, np.asarray(gt).reshape(-1, 4), keep, iou_thresh)
            pr += _img_pr_info(n_thresh, p, prop, rec)
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(pr[:, 0] > 0, pr[:, 1] / pr[:, 0], 0.0)
        recall = pr[:, 1] / max(n_faces, 1)
        out[setting] = voc_ap(recall, precision)
    return out


def load_ground_truth(eval_tools: str | Path) -> tuple[dict, dict]:
    """Read the official .mat files (eval_tools/ground_truth/). Image keys are
    "<event>/<image name without .jpg>"."""
    from scipy.io import loadmat

    root = Path(eval_tools)
    root = root / "ground_truth" if (root / "ground_truth").is_dir() else root
    mat = loadmat(root / "wider_face_val.mat")
    events, files, boxes = mat["event_list"], mat["file_list"], mat["face_bbx_list"]
    gts: dict[str, np.ndarray] = {}
    keeps: dict[str, dict[str, np.ndarray]] = {s: {} for s in SETTINGS}
    lists = {s: loadmat(root / f"wider_{s}_val.mat")["gt_list"] for s in SETTINGS}
    for i in range(len(events)):
        event = str(events[i][0][0])
        for j in range(len(files[i][0])):
            name = str(files[i][0][j][0][0])
            key = f"{event}/{name}"
            gts[key] = boxes[i][0][j][0].astype(np.float64)
            for s in SETTINGS:
                idx = lists[s][i][0][j][0]
                keeps[s][key] = (np.asarray(idx).reshape(-1) - 1).astype(np.int64)   # MATLAB is 1-based
    return gts, keeps
