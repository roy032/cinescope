"""Week 13 benchmark: video object removal with known truth.

    python p4_segmentation/bench_removal.py trailers/vtest.avi --start 30 --end 42
    # -> p4_segmentation/results/removal.json, removal_example.jpg

A person cut out of a *training* clip is pasted into a static-camera clip and
walks across it; the untouched clip is the ground truth for every pixel the
walker covers. Two walkers: a fast one (crosses ~400 px, so every covered
pixel is uncovered in some frame) and a slow one (~40 px, so the middle of
its path is never uncovered and must be inpainted spatially).

Rows:
  temporal plate (ours)      background_plate + fallback, given the exact masks
  per-frame Telea / NS       cv2.inpaint on every frame, exact masks
  per-frame Criminisi        ours, exact masks (every 10th frame: it is slow)
  end-to-end (ours)          the masks are *tracked* (YOLOv8n-seg boxes +
                             our GrabCut), then the temporal plate
PSNR over the true hole of every frame (dilated walker mask), SSIM over its
box; Criminisi's row is scored on its frames only, and so is a matching
"plate on the same frames" number.
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

from cinescope.io.video import VideoReader  # noqa: E402
from cinescope.segment.data import load_people, make_cutout, mask_box, mask_iou  # noqa: E402
from cinescope.segment.inpaint import criminisi, opencv_inpaint, psnr, ssim  # noqa: E402
from cinescope.segment.video_remove import remove_object, track_object  # noqa: E402


def walk(frames, cut, x0, x1, y_foot, height):
    s = height / cut.mask.shape[0]
    rgb = cv2.resize(cut.rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = cv2.resize(cut.mask.astype(np.float32), (rgb.shape[1], rgb.shape[0])) > 0.5
    out, masks = [], []
    for i, f in enumerate(frames):
        x = int(round(x0 + (x1 - x0) * i / max(1, len(frames) - 1)))
        y = y_foot - m.shape[0]
        g = f.copy()
        mm = np.zeros(f.shape[:2], bool)
        mm[y:y + m.shape[0], x:x + m.shape[1]] = m
        g[mm] = rgb[m]
        out.append(g)
        masks.append(mm)
    return out, masks


def score(results, truth, holes, idx):
    p = [psnr(results[i], truth[i], holes[i]) for i in idx]
    s = []
    for i in idx:
        x0, y0, x1, y1 = mask_box(holes[i]).astype(int)
        sl = (slice(max(0, y0 - 8), y1 + 8), slice(max(0, x0 - 8), x1 + 8))
        s.append(ssim(results[i][sl], truth[i][sl]))
    return round(float(np.mean(p)), 2), round(float(np.mean(s)), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--start", type=float, default=30)
    ap.add_argument("--end", type=float, default=42)
    ap.add_argument("--data", default="outputs/p4data")
    ap.add_argument("--out", default="p4_segmentation/results/removal.json")
    args = ap.parse_args()
    with VideoReader(args.clip) as vr:
        truth = [f.rgb for f in vr.frames(start_s=args.start, end_s=args.end)]
    h, w = truth[0].shape[:2]
    train = load_people(Path(args.data) / "people_train.npz")
    cands = [(f, k) for f in train for k in range(len(f.boxes)) if f.boxes[k, 3] - f.boxes[k, 1] >= 150]
    f, k = cands[len(cands) // 3]
    cut = make_cutout(f.rgb, f.masks[k], f.clip)
    height = int(0.17 * h)
    y_foot = int(0.62 * h)
    kernel = np.ones((5, 5), np.uint8)
    scenarios = {"fast walker": (int(0.15 * w), int(0.7 * w)), "slow walker": (int(0.45 * w), int(0.5 * w))}
    report = {"clip": Path(args.clip).name, "start": args.start, "end": args.end, "frames": len(truth),
              "walker_px_tall": height, "cutout_from": cut.source}
    example = None
    for name, (x0, x1) in scenarios.items():
        frames, masks = walk(truth, cut, x0, x1, y_foot, height)
        holes = [cv2.dilate(m.astype(np.uint8), kernel).astype(bool) for m in masks]
        every = list(range(len(frames)))
        sub = every[::10]
        rows = {}
        t = time.perf_counter()
        plate_out, rep = remove_object(frames, holes)
        ms = 1000 * (time.perf_counter() - t) / len(frames)
        rows["temporal plate (ours)"] = dict(zip(("psnr", "ssim"), score(plate_out, truth, holes, every), strict=True),
                                             ms_per_frame=round(ms, 1), never_seen_px=rep["never_seen_px"])
        rows["temporal plate (ours), Criminisi frames only"] = dict(zip(("psnr", "ssim"), score(plate_out, truth, holes, sub), strict=True))
        for meth in ("telea", "ns"):
            t = time.perf_counter()
            o = [opencv_inpaint(fr, hl, meth) for fr, hl in zip(frames, holes, strict=True)]
            ms = 1000 * (time.perf_counter() - t) / len(frames)
            rows[f"per-frame cv2 {'Telea' if meth == 'telea' else 'Navier-Stokes'}"] = dict(
                zip(("psnr", "ssim"), score(o, truth, holes, every), strict=True), ms_per_frame=round(ms, 1))
        t = time.perf_counter()
        crim = {i: criminisi(frames[i], holes[i]) for i in sub}
        ms = 1000 * (time.perf_counter() - t) / len(sub)
        full = [crim.get(i, frames[i]) for i in every]
        rows["per-frame Criminisi (ours), every 10th frame"] = dict(
            zip(("psnr", "ssim"), score(full, truth, holes, sub), strict=True), ms_per_frame=round(ms, 1))
        # end to end: tracked masks
        from cinescope.segment.models import PersonDetector
        from cinescope.segment.pipeline import get_segmenter

        det = PersonDetector()
        seg = get_segmenter("grabcut")
        t = time.perf_counter()
        tr = track_object(frames, mask_box(masks[0]), det, seg, dilate=4)
        e2e, rep2 = remove_object(frames, tr.masks)
        ms = 1000 * (time.perf_counter() - t) / len(frames)
        undilated = [cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) for m in tr.masks]
        rows["end-to-end: tracked GrabCut masks + plate (ours)"] = dict(
            zip(("psnr", "ssim"), score(e2e, truth, holes, every), strict=True), ms_per_frame=round(ms, 1),
            mask_iou=round(float(np.mean([mask_iou(a, b) for a, b in zip(undilated, masks, strict=True)])), 3),
            tracked_by_detection=round(float(tr.matched.mean()), 3))
        report[name] = rows
        print(name, json.dumps(rows, indent=1), flush=True)
        if example is None:
            i = len(frames) // 2
            x0b, y0b, x1b, y1b = mask_box(holes[i]).astype(int)
            pad = 40
            sl = (slice(max(0, y0b - pad), y1b + pad), slice(max(0, x0b - pad), x1b + pad))
            o_telea = opencv_inpaint(frames[i], holes[i], "telea")
            row = [frames[i][sl], plate_out[i][sl], o_telea[sl], truth[i][sl]]
            example = np.concatenate(row, 1)
    Path(args.out).write_text(json.dumps(report, indent=1))
    p = Path(args.out).with_name("removal_example.jpg")
    ex = cv2.resize(example, (example.shape[1] * 200 // example.shape[0], 200))
    cv2.imwrite(str(p), ex[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("->", args.out, p, "(columns: input, temporal plate, Telea, truth)")


if __name__ == "__main__":
    main()
