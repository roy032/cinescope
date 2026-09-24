"""Our face detector vs. a production one, on hand-labelled movie frames (week 8).

    python p2_detection/compare_faces.py p2_detection/labels/*.boxes.json \
        --ckpt p2_detection/checkpoints/centernet_r18.pt \
        --yunet models/face_detection_yunet_2023mar.onnx

The library baseline is OpenCV's YuNet (Wu et al., 2023): a small, fast
anchor-free face detector shipped with OpenCV's model zoo (download the .onnx
from github.com/opencv/opencv_zoo, models/face_detection_yunet). Both detectors
see the same frames; each clip is scored separately and together with the
COCO-style evaluator from week 6. Writes p2_detection/results/compare_faces.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.detect.cliplabels import frames_dir, load_labels, to_coco  # noqa: E402
from cinescope.detect.coco_eval import COCOEvaluator  # noqa: E402

HERE = Path(__file__).resolve().parent


class YuNet:
    name = "OpenCV YuNet"

    def __init__(self, model: str, score: float = 0.05):
        import cv2

        self.cv2, self.model, self.score = cv2, model, score
        self.det = None

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = rgb.shape[:2]
        if self.det is None:
            self.det = self.cv2.FaceDetectorYN.create(self.model, "", (w, h), self.score, 0.3, 5000)
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR))
        if faces is None:
            return np.zeros((0, 4)), np.zeros(0)
        return faces[:, :4].astype(np.float64), faces[:, 14].astype(np.float64)       # xywh, score


class Ours:
    name = "CenterNet-R18 (ours)"

    def __init__(self, ckpt: str, threshold: float = 0.05):
        from cinescope.detect.faces import FaceDetector

        self.det = FaceDetector(ckpt, threshold=threshold)

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.det.detect([rgb])[0]
        b = r["boxes"]
        return (np.c_[b[:, :2], b[:, 2:] - b[:, :2]] if len(b) else np.zeros((0, 4))), r["scores"]


def run(detector, clips: list[tuple[str, dict, Path]]) -> dict:
    from PIL import Image

    res, all_gt, all_dt, offset = {}, {"images": [], "annotations": [], "categories": []}, [], 0
    times = []
    for name, labels, fdir in clips:
        gt = to_coco(labels, ["face"], image_id_offset=offset)
        dts = []
        for im in gt["images"]:
            with Image.open(fdir / im["file_name"]) as f:
                rgb = np.asarray(f.convert("RGB"))
            if not times:
                detector(rgb)                                     # warm-up: first call loads/compiles
            t0 = time.perf_counter()
            boxes, scores = detector(rgb)
            times.append(time.perf_counter() - t0)
            dts += [{"image_id": im["id"], "category_id": 1, "bbox": list(map(float, b)), "score": float(s)}
                    for b, s in zip(boxes, scores, strict=True)]
        r = COCOEvaluator(gt).evaluate(dts)
        res[name] = {k: round(r[k], 4) for k in ("AP", "AP50", "AR100", "AP_small")}
        all_gt["images"] += gt["images"]
        all_gt["annotations"] += gt["annotations"]
        all_gt["categories"] = gt["categories"]
        all_dt += dts
        offset += len(gt["images"])
    r = COCOEvaluator(all_gt).evaluate(all_dt)
    res["all"] = {k: round(r[k], 4) for k in ("AP", "AP50", "AR100", "AP_small")}
    res["ms_per_frame"] = round(float(np.median(times)) * 1000, 1)
    return res


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--ckpt", help="our CenterNet checkpoint")
    ap.add_argument("--yunet", help="path to face_detection_yunet_*.onnx")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args(argv)

    clips = []
    for p in args.labels:
        lab = load_labels(p)
        clips.append((lab["name"], lab, frames_dir(p, lab)))
    detectors = ([Ours(args.ckpt)] if args.ckpt else []) + ([YuNet(args.yunet)] if args.yunet else [])
    if not detectors:
        ap.error("give --ckpt and/or --yunet")
    report = {d.name: run(d, clips) for d in detectors}
    names = [c[0] for c in clips] + ["all"]
    lines = ["| Detector | " + " | ".join(f"{n} AP50" for n in names) + " | all AP | ms/frame |",
             "|---|" + "---|" * (len(names) + 2)]
    for d, r in report.items():
        lines.append(f"| {d} | " + " | ".join(f"{r[n]['AP50']:.3f}" for n in names)
                     + f" | {r['all']['AP']:.3f} | {r['ms_per_frame']} |")
    table = "\n".join(lines)
    print(table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare_faces.json").write_text(json.dumps(report, indent=2))
    (out / "compare_faces.md").write_text(table + "\n")
    return report


if __name__ == "__main__":
    main()
