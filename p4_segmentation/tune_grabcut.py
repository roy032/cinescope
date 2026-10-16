"""Pick GrabCut's settings on *training-clip* composites (never the test clips).

    python p4_segmentation/tune_grabcut.py      # -> p4_segmentation/results/grabcut_dev.json

Composites: YOLOv8n-seg cut-outs from training frames pasted onto other
training frames (exact masks), prompted with the tight box. A small grid over
the crop margin, smoothness weight, superpixel size and minimum superpixel count; the best mean IoU
becomes `GrabCutParams`' defaults.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.segment.data import (  # noqa: E402
    composite,
    load_people,
    make_cutout,
    mask_box,
    mask_iou,
    place,
)
from cinescope.segment.grabcut import GrabCutParams, grabcut  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/p4data")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    frames = load_people(Path(args.data) / "people_train.npz")
    rng = np.random.default_rng(99)
    cuts = [make_cutout(f.rgb, f.masks[k], f.clip) for f in frames for k in range(len(f.boxes))
            if f.boxes[k, 3] - f.boxes[k, 1] >= 64]
    cases = []
    for _ in range(args.n):
        cut = cuts[rng.integers(len(cuts))]
        others = [g for g in frames if g.clip != cut.source]
        bg = others[rng.integers(len(others))].rgb
        placed, xy = place(cut, bg.shape, rng, (0.25, 0.8))
        cases.append(composite(bg, placed, xy))
    base = GrabCutParams()
    grid = {"margin": [0.25, 0.5], "gamma": [25.0, 50.0, 100.0], "sp_area": [30.0, 60.0],
            "min_segments": [16, 200]}
    rows = []
    for vals in itertools.product(*grid.values()):
        p = replace(base, **dict(zip(grid, vals, strict=True)))
        t = time.perf_counter()
        ious = [mask_iou(grabcut(img, mask_box(m), p), m) for img, m in cases]
        rows.append({**dict(zip(grid, vals, strict=True)), "miou": round(float(np.mean(ious)), 4),
                     "ms": round(1000 * (time.perf_counter() - t) / len(cases), 0)})
        print(rows[-1], flush=True)
    best = max(rows, key=lambda r: r["miou"])
    out = {"n": len(cases), "grid": rows, "best": best}
    Path("p4_segmentation/results/grabcut_dev.json").write_text(json.dumps(out, indent=1))
    print("best", best)


if __name__ == "__main__":
    main()
