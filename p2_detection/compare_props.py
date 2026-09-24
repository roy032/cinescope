"""Production detectors on movie frames: YOLO vs. RT-DETR on COCO props (week 8).

    pip install ultralytics
    python p2_detection/compare_props.py p2_detection/labels/*.boxes.json --models yolo11n.pt rtdetr-l.pt

Scores every labelled class that is also a COCO class (person, car, cell
phone, knife, bottle, ...). COCO has no "gun" class: guns are exactly the kind
of prop a film-specific model would need fine-tuning for. Weights download on
first use. Writes p2_detection/results/compare_props.{json,md}.
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


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--models", nargs="+", default=["yolo11n.pt", "rtdetr-l.pt"])
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args(argv)
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError:
        sys.exit("pip install ultralytics  (the library baselines for this comparison)")
    from PIL import Image

    report = {}
    for mpath in args.models:
        model = RTDETR(mpath) if "rtdetr" in Path(mpath).name.lower() else YOLO(mpath)
        coco_names = {v: k for k, v in model.names.items()}
        per_clip, times = {}, []
        for p in args.labels:
            lab = load_labels(p)
            classes = [c for c in lab["classes"] if c in coco_names]
            if not classes:
                continue
            gt = to_coco(lab, classes)
            fdir = frames_dir(p, lab)
            dts = []
            for im in gt["images"]:
                with Image.open(fdir / im["file_name"]) as f:
                    rgb = np.asarray(f.convert("RGB"))
                if not times:
                    model.predict(rgb[..., ::-1], verbose=False)       # warm-up
                t0 = time.perf_counter()
                r = model.predict(rgb[..., ::-1], conf=0.01, verbose=False)[0]   # ultralytics expects BGR arrays
                times.append(time.perf_counter() - t0)
                for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                   r.boxes.cls.cpu().numpy().astype(int), strict=True):
                    name = model.names[int(c)]
                    if name in classes:
                        dts.append({"image_id": im["id"], "category_id": classes.index(name) + 1,
                                    "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                                    "score": float(s)})
            ev = COCOEvaluator(gt).evaluate(dts)
            per_clip[lab["name"]] = {"AP": round(ev["AP"], 4), "AP50": round(ev["AP50"], 4),
                                     "per_class_AP": {classes[k - 1]: round(v, 4)
                                                      for k, v in ev["per_category_AP"].items() if v >= 0}}
        report[mpath] = {"clips": per_clip, "ms_per_frame": round(float(np.median(times)) * 1000, 1) if times else None}
    lines = ["| Model | " + " | ".join(f"{c} AP50" for c in next(iter(report.values()))["clips"]) + " | ms/frame |",
             "|---|" + "---|" * (len(next(iter(report.values()))["clips"]) + 1)]
    for m, r in report.items():
        lines.append(f"| {m} | " + " | ".join(f"{v['AP50']:.3f}" for v in r["clips"].values()) + f" | {r['ms_per_frame']} |")
    table = "\n".join(lines)
    print(table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare_props.json").write_text(json.dumps(report, indent=2))
    (out / "compare_props.md").write_text(table + "\n")
    return report


if __name__ == "__main__":
    main()
