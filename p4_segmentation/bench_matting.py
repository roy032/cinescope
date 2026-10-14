"""Week 13 benchmark: closed-form matting vs. pymatting, on composites with known alpha.

    python p4_segmentation/bench_matting.py            # needs outputs/p4data/people_test.npz
    # -> p4_segmentation/results/matting.json

Each composite: a test person's pixels, extended outward with the nearest
person colour, are the foreground F; a frame of a
different test clip is the background B, and alpha is a soft matte built from
the person's YOLOv8n-seg mask (blurred edge + anti-aliased hair strands,
`data.soft_alpha`). I = alpha F + (1 - alpha) B, so alpha and F are known
exactly. Every method gets the same trimap, made from the binary mask
alpha > 0.5 the way the cut-out tool makes it (erode/dilate by 3% of the
person's height). Scores are over the unknown band: SAD (sum |error| / 1000)
and MSE. Nothing here was tuned: eps = 1e-7 and 3x3 windows as in the paper.

Foreground colours are scored too (MSE of F, 0..1 units, where true alpha > 0.05,
given the true alpha so only the colour estimate is compared).
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

from cinescope.segment.data import alpha_composite, load_people, mask_box, soft_alpha  # noqa: E402
from cinescope.segment.matting import (  # noqa: E402
    closed_form_alpha,
    estimate_foreground,
    mse,
    sad,
    trimap_from_mask,
)


def make_cases(frames, n, rng, pad=24, clean=False):
    inst = [(f, k) for f in frames for k in range(len(f.boxes)) if f.boxes[k, 3] - f.boxes[k, 1] >= 64]
    cases = []
    for i in range(n):
        f, k = inst[(i * 7) % len(inst)]
        others = [g for g in frames if g.clip != f.clip]
        bg = others[rng.integers(len(others))].rgb
        bg = cv2.resize(bg, (f.rgb.shape[1], f.rgb.shape[0]), interpolation=cv2.INTER_AREA)
        from scipy import ndimage as ndi

        lab, nl = ndi.label(f.masks[k])
        mask = lab == (1 + int(np.bincount(lab.ravel())[1:].argmax())) if nl > 1 else f.masks[k]
        alpha = soft_alpha(mask, rng)
        h, w = mask.shape
        x0, y0, x1, y1 = mask_box(alpha > 0.01).astype(int)
        sl = (slice(max(0, y0 - pad), min(h, y1 + pad)), slice(max(0, x0 - pad), min(w, x1 + pad)))
        # F outside the mask = colour of the nearest person pixel, so strands and
        # blurred edges carry the person's colours (not the source frame's background).
        # clean: extend from 6 px inside the mask, so F has no edges of its own
        # in the band (the colour change there is exactly the alpha change)
        src = ndi.binary_erosion(mask, np.ones((3, 3), bool), 6) if clean else mask
        if not src.any():
            src = mask
        _, (iy, ix) = ndi.distance_transform_edt(~src, return_indices=True)
        fg = f.rgb[iy, ix][sl].astype(np.float64)
        a = alpha[sl]
        img = alpha_composite(fg, bg[sl].astype(np.float64), a)
        cases.append({"img": img, "alpha": a, "fg": fg / 255.0, "mask": a > 0.5, "clip": f.clip})
    return cases


def score(cases, methods, pymatting):
    res = {k: {"sad": [], "mse": [], "ms": []} for k in methods}
    fg_res = {"blur-fusion (ours)": {"mse": [], "ms": []}, "image as F (no estimation)": {"mse": []}}
    if pymatting:
        fg_res["pymatting ML (multi-level)"] = {"mse": [], "ms": []}
    for c in cases:
        tri = trimap_from_mask(c["mask"])
        unk = tri == 0.5
        for name, fn in methods.items():
            t = time.perf_counter()
            a = fn(c["img"], tri, c["mask"])
            res[name]["ms"].append(1000 * (time.perf_counter() - t))
            a = np.clip(a, 0, 1)
            res[name]["sad"].append(sad(a, c["alpha"], unk))
            res[name]["mse"].append(mse(a, c["alpha"], unk))
        sel = c["alpha"] > 0.05
        t = time.perf_counter()
        f = estimate_foreground(c["img"], c["alpha"])
        fg_res["blur-fusion (ours)"]["ms"].append(1000 * (time.perf_counter() - t))
        fg_res["blur-fusion (ours)"]["mse"].append(float(((f - c["fg"]) ** 2)[sel].mean()))
        fg_res["image as F (no estimation)"]["mse"].append(float(((c["img"] / 255.0 - c["fg"]) ** 2)[sel].mean()))
        if pymatting:
            t = time.perf_counter()
            f = pymatting.estimate_foreground_ml(c["img"] / 255.0, c["alpha"])
            fg_res["pymatting ML (multi-level)"]["ms"].append(1000 * (time.perf_counter() - t))
            fg_res["pymatting ML (multi-level)"]["mse"].append(float(((f - c["fg"]) ** 2)[sel].mean()))

    def summ(d):
        return {k: round(float(np.mean(v)), 5 if k == "mse" else 3) for k, v in d.items() if v}

    return {"n": len(cases), "mean_unknown_px": int(np.mean([(trimap_from_mask(c["mask"]) == 0.5).sum() for c in cases])),
            "alpha": {k: summ(v) for k, v in res.items()},
            "foreground": {k: summ(v) for k, v in fg_res.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="outputs/p4data")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="p4_segmentation/results/matting.json")
    args = ap.parse_args()
    frames = load_people(Path(args.data) / "people_test.npz")
    cases = make_cases(frames, args.n, np.random.default_rng(13))

    methods = {
        "binary mask": lambda img, tri, m: m.astype(np.float64),
        "feathered mask (Gaussian 1.5 px)": lambda img, tri, m: cv2.GaussianBlur(m.astype(np.float64), (0, 0), 1.5),
        "closed-form (ours)": lambda img, tri, m: closed_form_alpha(img, tri),
    }
    try:
        import pymatting

        methods["pymatting closed-form"] = lambda img, tri, m: pymatting.estimate_alpha_cf(img / 255.0, tri)
        methods["pymatting KNN"] = lambda img, tri, m: pymatting.estimate_alpha_knn(img / 255.0, tri)
        methods["pymatting LKM"] = lambda img, tri, m: pymatting.estimate_alpha_lkm(img / 255.0, tri)
        # numba compiles on first use; do it outside the timing
        c0 = cases[0]
        t0 = trimap_from_mask(c0["mask"])
        for k in ("pymatting closed-form", "pymatting KNN", "pymatting LKM"):
            methods[k](c0["img"], t0, c0["mask"])
        pymatting.estimate_foreground_ml(c0["img"] / 255.0, c0["alpha"])
    except ImportError:
        pymatting = None
        print("pymatting not installed: only our methods are scored")

    out = {"realistic": score(cases, methods, pymatting),
           "clean": score(make_cases(frames, args.n, np.random.default_rng(13), clean=True), methods, pymatting)}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
