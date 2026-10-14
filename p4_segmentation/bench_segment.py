"""Week 12 benchmark: box-prompted person masks, from scratch vs. production.

    python p4_segmentation/make_people.py --role test trailers/goldeneye.mp4 trailers/vtest.avi trailers/people-detection.mp4
    python p4_segmentation/bench_segment.py
    # -> p4_segmentation/results/segment.json, segment_examples.jpg

Two scores, because there are no hand-made masks:

  vs. SAM 2     every confident, whole YOLOv8n-seg person box in the test
                frames is a prompt; SAM 2 tiny's mask is the reference. This
                measures agreement with a strong model on real footage, not
                accuracy (SAM 2 is scored 1.0 by definition).
  exact truth   composites: a test person cut out (the pixels where SAM 2 and
                YOLOv8n-seg *both* say person, largest piece) and pasted onto a
                frame of a *different* test clip. The pasted mask is the truth
                exactly, so every method, SAM 2 included, is scored on it. The
                prompt is the pasted mask's tight box.

Methods: our GrabCut (superpixel + band graph cut, own max-flow), cv2.grabCut
on the same crop, our U-Net, MobileSAM, SAM 2 tiny, and YOLOv8n-seg (not
promptable: its mask whose box best matches the prompt). Times are wall-clock
per prompt on one CPU thread; SAM-family models pay their image encoder once
per frame, so on frames with several prompts their per-prompt time is shared.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
from cinescope.segment.grabcut import grabcut, grabcut_opencv  # noqa: E402
from cinescope.segment.models import (  # noqa: E402
    PersonDetector,
    SamSegmenter,
    YoloBoxSegmenter,
    set_threads,
)


def build_methods(args):
    methods = {
        "GrabCut (ours)": lambda rgb, boxes: [grabcut(rgb, b) for b in boxes],
        "cv2.grabCut": lambda rgb, boxes: [grabcut_opencv(rgb, b) for b in boxes],
    }
    if Path(args.unet).exists():
        from cinescope.segment.unet import UNetSegmenter

        methods["U-Net (ours)"] = UNetSegmenter(args.unet)
    else:
        print(f"no U-Net checkpoint at {args.unet}; skipping it")
    methods["YOLOv8n-seg"] = YoloBoxSegmenter(PersonDetector(threads=args.threads))
    methods["MobileSAM"] = SamSegmenter("mobile_sam.pt", threads=args.threads)
    methods["SAM 2 tiny"] = SamSegmenter("sam2_t.pt", threads=args.threads)
    return methods


def run(methods, items, warm=True):
    """items: list of (rgb, boxes). Returns {method: (list of mask lists, seconds per item)}."""
    out = {}
    for name, fn in methods.items():
        if warm and items:
            fn(items[0][0], items[0][1][:1])
        masks, secs = [], []
        for rgb, boxes in items:
            t = time.perf_counter()
            masks.append(fn(rgb, boxes))
            secs.append(time.perf_counter() - t)
        out[name] = (masks, secs)
        n_boxes = sum(len(b) for _, b in items)
        print(f"  {name}: {1000 * sum(secs) / max(1, n_boxes):.0f} ms/prompt", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="outputs/p4data")
    ap.add_argument("--unet", default="p4_segmentation/checkpoints/unet_person.pt")
    ap.add_argument("--n-composites", type=int, default=60)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="p4_segmentation/results/segment.json")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="re-run only these methods (SAM 2 still runs on the real frames as the "
                         "reference) and merge their rows into an existing --out")
    args = ap.parse_args()
    set_threads(args.threads)
    frames = load_people(Path(args.data) / "people_test.npz")
    methods = build_methods(args)
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods or k == "SAM 2 tiny"}

    # ---- part A: agreement with SAM 2 on real test frames
    print(f"vs. SAM 2: {len(frames)} frames, {sum(len(f.boxes) for f in frames)} prompts")
    items = [(f.rgb, f.boxes) for f in frames]
    res_a = run(methods, items)
    ref = res_a["SAM 2 tiny"][0]
    table_a = {}
    for name, (masks, secs) in res_a.items():
        per_clip: dict[str, list[float]] = {}
        for f, ms, rs in zip(frames, masks, ref, strict=True):
            for m, r in zip(ms, rs, strict=True):
                per_clip.setdefault(f.clip, []).append(mask_iou(m, r))
        all_iou = [v for vs in per_clip.values() for v in vs]
        n = sum(len(f.boxes) for f in frames)
        table_a[name] = {"miou": round(float(np.mean(all_iou)), 4),
                         "per_clip": {k: round(float(np.mean(v)), 4) for k, v in per_clip.items()},
                         "ms_per_prompt": round(1000 * sum(secs) / n, 1),
                         "ms_per_frame": round(1000 * float(np.mean(secs)), 1)}

    # ---- part B: exact truth on composites
    rng = np.random.default_rng(12)
    cuts = []
    for f, rs in zip(frames, ref, strict=True):
        for k, r in enumerate(rs):
            both = r & f.masks[k]
            if both.sum() > 0 and f.boxes[k, 3] - f.boxes[k, 1] >= 64:
                cuts.append(make_cutout(f.rgb, both, f.clip))
    comps = []
    for i in range(args.n_composites):
        cut = cuts[(i * len(cuts)) // args.n_composites]     # spread over all clips' people
        others = [g for g in frames if g.clip != cut.source]
        bg = others[rng.integers(len(others))].rgb
        placed, xy = place(cut, bg.shape, rng, height_frac=(0.25, 0.8))
        img, mask = composite(bg, placed, xy)
        comps.append((img, mask, cut.source))
    print(f"exact truth: {len(comps)} composites from {len(cuts)} cut-outs")
    items_b = [(img, mask_box(m)[None]) for img, m, _ in comps]
    res_b = run({k: v for k, v in methods.items() if not args.methods or k in args.methods}, items_b)
    table_b = {}
    for name, (masks, secs) in res_b.items():
        ious = [mask_iou(ms[0], m) for ms, (_, m, _) in zip(masks, comps, strict=True)]
        table_b[name] = {"miou": round(float(np.mean(ious)), 4),
                         "median_iou": round(float(np.median(ious)), 4),
                         "iou_below_0.5": int(np.sum(np.asarray(ious) < 0.5)),
                         "per_source_clip": {c: round(float(np.mean([v for v, (_, _, src) in zip(ious, comps, strict=True)
                                                                    if src == c])), 4)
                                             for c in sorted({src for _, _, src in comps})},
                         "ms_per_frame": round(1000 * float(np.mean(secs)), 1)}

    if args.methods and Path(args.out).exists():
        old = json.loads(Path(args.out).read_text())
        old["vs_sam2"].update({k: v for k, v in table_a.items() if k in args.methods})
        old["exact"].update(table_b)
        Path(args.out).write_text(json.dumps(old, indent=1))
        print(json.dumps({k: old[k] for k in ("vs_sam2", "exact")}, indent=1))
        return
    out = {"vs_sam2": table_a, "exact": table_b,
           "n_frames": len(frames), "n_prompts": int(sum(len(f.boxes) for f in frames)),
           "n_composites": len(comps), "n_cutouts": len(cuts),
           "threads": args.threads}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    _examples(res_b, comps, Path(args.out).with_name("segment_examples.jpg"))


def _examples(res, comps, path, n=4):
    """A small grid: composite crop, truth, then each method's mask (as cut-outs on grey)."""
    import cv2

    rows = []
    for i in np.linspace(0, len(comps) - 1, n).round().astype(int):
        img, m, _ = comps[i]
        x0, y0, x1, y1 = mask_box(m).astype(int)
        pad = 10
        sl = (slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad))
        tiles = [img[sl]]
        for msk in [m] + [res[k][0][i][0] for k in res]:
            t = np.where(msk[sl][..., None], img[sl], 128).astype(np.uint8)
            tiles.append(t)
        row = np.concatenate(tiles, 1)
        rows.append(cv2.resize(row, (row.shape[1] * 160 // row.shape[0], 160)))
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
    grid = np.concatenate(rows, 0)
    if grid.shape[1] > 1400:
        grid = cv2.resize(grid, (1400, grid.shape[0] * 1400 // grid.shape[1]))
    cv2.imwrite(str(path), grid[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("->", path, "(columns: composite, truth, " + ", ".join(res) + ")")


if __name__ == "__main__":
    main()
