"""Week 13 benchmark: spatial inpainting — Criminisi (ours) vs. cv2.inpaint.

    python p4_segmentation/bench_inpaint.py         # needs outputs/p4data/people_{train,test}.npz
    # -> p4_segmentation/results/inpaint.json, inpaint_examples.jpg

Holes are person-shaped (the mask of a *training*-clip person, rescaled to
15-30% of the frame height, dilated by 3 px as a removal mask would be) and
punched into real *test* frames at random places. The frame's original pixels
are the ground truth. Scores: PSNR over the hole pixels, SSIM over the hole's
bounding box plus 8 px (SSIM over the whole frame would be ~1 for everything),
and wall-clock time on one CPU thread. Patch 9 and search 60 px are the
defaults, not tuned on these frames.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from cinescope.segment.data import load_people, make_cutout, place  # noqa: E402
from cinescope.segment.inpaint import criminisi, opencv_inpaint, psnr, ssim  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="outputs/p4data")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="p4_segmentation/results/inpaint.json")
    args = ap.parse_args()
    data = Path(args.data)
    test = load_people(data / "people_test.npz")
    train = load_people(data / "people_train.npz")
    shapes = [make_cutout(f.rgb, f.masks[k]) for f in train for k in range(len(f.boxes))
              if f.boxes[k, 3] - f.boxes[k, 1] >= 64]
    rng = np.random.default_rng(14)
    methods = {
        "Criminisi (ours)": lambda im, h: criminisi(im, h),
        "cv2 Telea": lambda im, h: opencv_inpaint(im, h, "telea"),
        "cv2 Navier-Stokes": lambda im, h: opencv_inpaint(im, h, "ns"),
    }
    res = {k: {"psnr": [], "ssim": [], "ms": []} for k in methods}
    examples = []
    for i in range(args.n):
        img = test[(i * 5) % len(test)].rgb
        shape, (x, y) = place(shapes[rng.integers(len(shapes))], img.shape, rng, height_frac=(0.15, 0.3))
        hole = np.zeros(img.shape[:2], bool)
        hh, ww = shape.mask.shape
        hole[y:y + hh, x:x + ww] = shape.mask
        hole = cv2.dilate(hole.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        damaged = img.copy()
        damaged[hole] = 0
        ys, xs = np.nonzero(hole)
        sl = (slice(max(0, ys.min() - 8), ys.max() + 9), slice(max(0, xs.min() - 8), xs.max() + 9))
        row = [img[sl], damaged[sl]]
        for name, fn in methods.items():
            t = time.perf_counter()
            out = fn(damaged, hole)
            res[name]["ms"].append(1000 * (time.perf_counter() - t))
            res[name]["psnr"].append(psnr(out, img, hole))
            res[name]["ssim"].append(ssim(out[sl], img[sl]))
            row.append(out[sl])
        if len(examples) < 4:
            r = np.concatenate(row, 1)
            examples.append(cv2.resize(r, (r.shape[1] * 140 // r.shape[0], 140)))
        print(i, {k: round(v["psnr"][-1], 2) for k, v in res.items()}, flush=True)
    out = {"n": args.n,
           "methods": {k: {m: round(float(np.mean(v)), 3 if m != "ms" else 0) for m, v in d.items()}
                       for k, d in res.items()},
           "criminisi_wins_psnr": int(np.sum(np.asarray(res["Criminisi (ours)"]["psnr"])
                                             > np.asarray(res["cv2 Telea"]["psnr"])))}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    wmax = max(e.shape[1] for e in examples)
    grid = np.concatenate([np.pad(e, ((0, 0), (0, wmax - e.shape[1]), (0, 0))) for e in examples], 0)
    if grid.shape[1] > 1200:
        grid = cv2.resize(grid, (1200, grid.shape[0] * 1200 // grid.shape[1]))
    p = Path(args.out).with_name("inpaint_examples.jpg")
    cv2.imwrite(str(p), grid[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("->", p, "(columns: original, hole, Criminisi, Telea, NS)")


if __name__ == "__main__":
    main()
