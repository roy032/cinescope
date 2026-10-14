"""Object Keypoint Similarity and COCO keypoint AP, from the definitions.

OKS plays the role IoU plays for boxes: a number in [0, 1] saying how well a
predicted skeleton matches a labelled one,

    OKS = mean over labelled keypoints i of  exp(-d_i^2 / (2 s^2 k_i^2))

with d_i the pixel distance between prediction and label, s^2 the object's
area (so s is its scale in pixels) and k_i = 2 sigma_i the per-keypoint
tolerance. sigma_i comes from how much human annotators disagree about that
keypoint, so an error the size of annotator noise costs little; being 2 cm
off on an eye costs more than 2 cm off on a hip. Only keypoints labelled
visible or occluded-but-placed (v > 0) count.

Keypoint AP is then the box AP protocol with OKS in place of IoU: thresholds
0.50:0.05:0.95, up to 20 detections per image, area ranges all / medium /
large (COCO does not label keypoints on small people). The matching and
precision/recall bookkeeping are shared with cinescope.detect.coco_eval; only
the similarity changes. tests/test_week18.py checks the numbers against
pycocotools' COCOeval(iouType="keypoints") on random data.
"""
from __future__ import annotations

import numpy as np

from ..detect.coco_eval import COCOEvaluator, EvalParams
from . import COCO_SIGMAS


def oks(pred: np.ndarray, gt: np.ndarray, vis: np.ndarray, area: float,
        sigmas: np.ndarray = COCO_SIGMAS) -> float:
    """OKS of one prediction (K, 2) against one labelled pose (K, 2) with
    visibility flags (K,) and object area (pixels^2). NaN when nothing is labelled."""
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    m = np.asarray(vis) > 0
    if not m.any():
        return float("nan")
    var = (2 * sigmas) ** 2
    d2 = ((pred - gt) ** 2).sum(-1)
    e = d2 / var / (area + np.spacing(1)) / 2
    return float(np.exp(-e[m]).mean())


def oks_matrix(dts: list[dict], gts: list[dict], sigmas: np.ndarray = COCO_SIGMAS) -> np.ndarray:
    """(D, G) OKS between COCO-format detections and ground truths.

    A ground truth with no labelled keypoints (a crowd region or an ignored
    person) scores a detection by how far its keypoints fall outside the
    ground-truth box grown to 3x its size — pycocotools' rule, kept so the
    numbers agree; such ground truths are ignored in matching anyway."""
    var = (2 * sigmas) ** 2
    out = np.zeros((len(dts), len(gts)))
    if not dts or not gts:
        return out
    d = np.array([dt["keypoints"] for dt in dts], np.float64).reshape(len(dts), -1, 3)
    for j, g in enumerate(gts):
        gk = np.asarray(g["keypoints"], np.float64).reshape(-1, 3)
        vis = gk[:, 2] > 0
        if vis.any():
            dx = d[:, :, 0] - gk[None, :, 0]
            dy = d[:, :, 1] - gk[None, :, 1]
        else:
            x, y, w, h = g["bbox"]
            x0, x1, y0, y1 = x - w, x + 2 * w, y - h, y + 2 * h
            dx = np.maximum(0, x0 - d[:, :, 0]) + np.maximum(0, d[:, :, 0] - x1)
            dy = np.maximum(0, y0 - d[:, :, 1]) + np.maximum(0, d[:, :, 1] - y1)
        e = (dx ** 2 + dy ** 2) / var / (g["area"] + np.spacing(1)) / 2
        if vis.any():
            e = e[:, vis]
        out[:, j] = np.exp(-e).mean(1)
    return out


def keypoint_params() -> EvalParams:
    return EvalParams(max_dets=(20,), area_ranges={"all": (0.0, 1e10), "medium": (32.0 ** 2, 96.0 ** 2),
                                                   "large": (96.0 ** 2, 1e10)})


class KeypointEvaluator(COCOEvaluator):
    """COCO keypoint evaluation.

        ev = KeypointEvaluator(gt)    # annotations carry "keypoints" (3K list), "bbox", "area", "num_keypoints"
        ev.evaluate(dets)             # [{"image_id", "category_id", "keypoints", "score"}]

    Returns AP, AP50, AP75, AP_medium, AP_large, AR20, AR_medium, AR_large.
    """

    def __init__(self, gt: dict, params: EvalParams | None = None, sigmas: np.ndarray = COCO_SIGMAS):
        gt = dict(gt)
        anns = []
        for a in gt["annotations"]:
            a = dict(a)
            nk = a.get("num_keypoints", int((np.asarray(a["keypoints"]).reshape(-1, 3)[:, 2] > 0).sum()))
            # pycocotools: a person without labelled keypoints cannot be scored, so it is ignored
            a["ignore"] = int(bool(a.get("ignore", 0)) or bool(a.get("iscrowd", 0)) or nk == 0)
            anns.append(a)
        gt["annotations"] = anns
        super().__init__(gt, params or keypoint_params())
        self.similarity = lambda dts, gts: oks_matrix(dts, gts, sigmas)

    def evaluate(self, dets: list[dict]) -> dict:
        # a detection's area (used only for the area ranges) is that of its keypoints' extent
        prepared = []
        for d in dets:
            k = np.asarray(d["keypoints"], np.float64).reshape(-1, 3)
            x0, x1, y0, y1 = k[:, 0].min(), k[:, 0].max(), k[:, 1].min(), k[:, 1].max()
            prepared.append({**d, "bbox": [x0, y0, x1 - x0, y1 - y0]})
        return super().evaluate(prepared)
