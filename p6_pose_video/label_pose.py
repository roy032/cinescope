"""Render person crops with a coordinate grid, for keypoint labelling by eye.

    python p6_pose_video/label_pose.py render goldeneye --every 2.5 --out outputs/p6/label_pose
    # look at each crop, write p6_pose_video/labels/<clip>_pose_grid.json (grid units, see below)
    python p6_pose_video/label_pose.py convert p6_pose_video/labels/goldeneye_pose_grid.json \
        --index outputs/p6/label_pose/index.json --out p6_pose_video/labels/goldeneye_pose.json

People are found with the plain YOLOv8n *detector* (boxes only), never the
pose model, so the labeller does not see any predicted skeleton. Each crop is
the box padded 15%, scaled to 480 px tall, with a grid in 0-100 units on both
axes (lines every 10, ticks every 5). A label gives each of the 17 COCO
keypoints as [gx, gy, v] in grid units, v = 2 visible, 1 hidden but
placeable, 0 not labelled; `convert` maps them back to frame pixels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.io.video import VideoReader  # noqa: E402
from cinescope.video.clips import clip_name, resolve  # noqa: E402

PAD, OUT_H = 0.15, 480


def render(args):
    import torch

    torch.set_num_threads(2)
    from cinescope.pose.yolo import YoloPose

    det = YoloPose(pose=False, conf=0.5)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    path = resolve(args.clip)
    with VideoReader(path) as vr:
        frames = list(vr.frames(sample_fps=1.0 / args.every))
    for f in frames:
        p = det([f.rgb])[0]
        H, W = f.rgb.shape[:2]
        for b, s in zip(p.boxes, p.scores, strict=True):
            if b[3] - b[1] < args.min_height:
                continue
            bw, bh = b[2] - b[0], b[3] - b[1]
            x0, y0 = max(0, b[0] - PAD * bw), max(0, b[1] - PAD * bh)
            x1, y1 = min(W, b[2] + PAD * bw), min(H, b[3] + PAD * bh)
            crop = Image.fromarray(f.rgb[int(y0):int(y1), int(x0):int(x1)])
            sc = OUT_H / crop.height
            crop = crop.resize((max(1, round(crop.width * sc)), OUT_H), Image.BICUBIC)
            canvas = Image.new("RGB", (crop.width + 40, crop.height + 30), (255, 255, 255))
            canvas.paste(crop, (30, 0))
            dr = ImageDraw.Draw(canvas)
            for g in range(0, 101, 5):
                gx = 30 + g / 100 * (crop.width - 1)
                gy = g / 100 * (crop.height - 1)
                col = (255, 255, 0) if g % 10 == 0 else (0, 255, 255)
                width = 1
                if g % 10 == 0:
                    dr.text((gx - 6, crop.height + 4), str(g), fill=(0, 0, 0))
                    dr.text((2, gy - 5), str(g), fill=(0, 0, 0))
                dr.line([(gx, 0), (gx, crop.height - 1)], fill=col, width=width)
                dr.line([(30, gy), (30 + crop.width - 1, gy)], fill=col, width=width)
            pid = f"{clip_name(path)}_{f.t:06.2f}_{len(index):03d}"
            canvas.save(out / f"{pid}.jpg", quality=90)
            index.append({"id": pid, "clip": clip_name(path), "t": round(f.t, 3), "frame_wh": [W, H],
                          "box": [round(float(v), 1) for v in b], "score": round(float(s), 3),
                          "window": [float(x0), float(y0), float(x1), float(y1)]})
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"{len(index)} people from {len(frames)} frames -> {out}")


def convert(args):
    grid = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    index = {e["id"]: e for e in json.loads(Path(args.index).read_text(encoding="utf-8"))}
    people = []
    for pid, lab in grid["people"].items():
        e = index[pid]
        x0, y0, x1, y1 = e["window"]
        k = np.asarray(lab["kpts"], np.float64).reshape(17, 3)
        xy = np.stack([x0 + k[:, 0] / 100 * (x1 - x0), y0 + k[:, 1] / 100 * (y1 - y0)], 1)
        kp = np.concatenate([np.round(xy, 1), k[:, 2:]], 1)
        kp[k[:, 2] == 0, :2] = 0
        vis = kp[:, 2] > 0
        # box: the detector box, grown to cover every labelled keypoint
        bx0, by0, bx1, by1 = e["box"]
        if vis.any():
            bx0, by0 = min(bx0, kp[vis, 0].min()), min(by0, kp[vis, 1].min())
            bx1, by1 = max(bx1, kp[vis, 0].max()), max(by1, kp[vis, 1].max())
        people.append({"id": pid, "clip": e["clip"], "t": e["t"], "frame_wh": e["frame_wh"],
                       "bbox": [bx0, by0, bx1 - bx0, by1 - by0], "keypoints": kp.ravel().tolist(),
                       "num_keypoints": int(vis.sum()), "note": lab.get("note", "")})
    out = {"name": grid.get("name", "pose"), "how": grid.get("how", ""), "people": people}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(people)} people, {sum(p['num_keypoints'] for p in people)} keypoints -> {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("clip")
    r.add_argument("--every", type=float, default=2.5, help="seconds between sampled frames")
    r.add_argument("--min-height", type=float, default=150)
    r.add_argument("--out", default="outputs/p6/label_pose")
    c = sub.add_parser("convert")
    c.add_argument("labels")
    c.add_argument("--index", default="outputs/p6/label_pose/index.json")
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    render(args) if args.cmd == "render" else convert(args)


if __name__ == "__main__":
    main()
