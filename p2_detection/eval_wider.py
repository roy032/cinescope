"""Evaluate a face detector on the WIDER FACE validation set.

    python p2_detection/eval_wider.py --wider D:/datasets/WIDER --ckpt p2_detection/checkpoints/centernet_r18.pt
    python p2_detection/eval_wider.py ... --eval-tools D:/datasets/WIDER/eval_tools    # official Easy/Medium/Hard

With the official toolkit's ground-truth .mat files, reports Easy / Medium /
Hard AP computed the toolkit's way (cinescope/detect/wider_eval.py). Without
them, falls back to AP@0.5 over all valid faces — a stricter, *different*
number; the report says which one it is.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.detect.boxes import xywh_to_xyxy, xyxy_to_xywh  # noqa: E402
from cinescope.detect.coco_eval import ap_at_iou  # noqa: E402
from cinescope.detect.wider import parse_annotations  # noqa: E402
from cinescope.detect.wider_eval import evaluate, load_ground_truth  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wider", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-tools", help="folder with wider_face_val.mat and wider_{easy,medium,hard}_val.mat")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--threshold", type=float, default=0.02, help="keep low-score boxes: AP needs the whole PR curve")
    ap.add_argument("--limit", type=int, help="first N images (smoke test)")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args(argv)

    from PIL import Image

    from cinescope.detect.faces import FaceDetector

    root = Path(args.wider)
    items = parse_annotations(root / "wider_face_split" / "wider_face_val_bbx_gt.txt")[: args.limit]
    det = FaceDetector(args.ckpt, threshold=args.threshold, max_side=args.max_side)
    preds, times = {}, []
    for k, it in enumerate(items):
        with Image.open(root / "WIDER_val" / "images" / it.path) as im:
            rgb = np.asarray(im.convert("RGB"))
        t0 = time.perf_counter()
        r = det.detect([rgb])[0]
        times.append(time.perf_counter() - t0)
        key = it.path[:-4] if it.path.lower().endswith(".jpg") else it.path
        preds[key] = np.c_[xyxy_to_xywh(r["boxes"]), r["scores"]] if len(r["boxes"]) else np.zeros((0, 5))
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(items)} images, {np.mean(times) * 1000:.0f} ms/image")

    report = {"checkpoint": args.ckpt, "images": len(items), "ms_per_image": round(float(np.mean(times)) * 1000, 1),
              "max_side": args.max_side}
    if args.eval_tools:
        gts, keeps = load_ground_truth(args.eval_tools)
        if args.limit:
            gts = {k: v for k, v in gts.items() if k in preds}
        report["protocol"] = "official WIDER FACE (easy/medium/hard)"
        report.update(evaluate(preds, gts, keeps))
    else:
        g = [xywh_to_xyxy(it.boxes) for it in items]
        keys = [it.path[:-4] for it in items]
        d = [xywh_to_xyxy(preds[k][:, :4]) for k in keys]
        s = [preds[k][:, 4] for k in keys]
        report["protocol"] = "AP@0.5, all valid faces (NOT the official easy/medium/hard numbers)"
        report["ap50_all_faces"] = ap_at_iou(g, d, s, 0.5)["ap"]
    print(json.dumps(report, indent=2))
    out = Path(args.out) / Path(args.ckpt).stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "wider_val.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
