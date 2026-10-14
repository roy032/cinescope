"""Distillation data for the pose network: YOLOv8n-pose labels people in clips.

    python p6_pose_video/build_pose_data.py --out outputs/p6/pose_train.npz            # training clips
    python p6_pose_video/build_pose_data.py --clips goldeneye people-detection --fps 2 --out outputs/p6/pose_test.npz

Frames are sampled at `--fps`; each person the teacher finds (box height >=
`--min-height` px, at least 6 keypoints with confidence >= 0.5) becomes one
training crop. The crop window is the person box padded the usual 25%,
enlarged a further `ENLARGE`x so training can scale and rotate it without
running into black borders, stored at 168 x 126 px. Keypoints are stored in
crop coordinates with the teacher's confidences; the loss later uses
confidence >= 0.5 as "labelled".

For the evaluation set (`--keep-frames`) the frames' teacher output (boxes,
scores, keypoints) is stored per frame instead, so the from-scratch network can
be run top-down on the teacher's boxes and on frames it never saw.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.io.video import VideoReader  # noqa: E402
from cinescope.pose.heatmaps import box_to_center_scale, crop_matrix, transform_points  # noqa: E402
from cinescope.video.clips import clip_name, resolve  # noqa: E402

# Clips the pose network trains on. Held out (never in training): goldeneye
# (film; by-eye keypoint labels) and people-detection (same surveillance domain
# as most training clips, to separate "unseen frames" from "unseen domain").
TRAIN_CLIPS = ["vtest", "classroom", "face-demographics-walking", "head-pose-face-detection-female-and-male",
               "one-by-one-person-detection", "person-bicycle-car-detection", "store-aisle-detection",
               "worker-zone-detection", "car-detection"]
STORE_HW = (168, 126)
ENLARGE = 1.3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", default=TRAIN_CLIPS)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--min-height", type=float, default=48)
    ap.add_argument("--max-width", type=int, default=1280, help="downscale wider frames first")
    ap.add_argument("--keep-frames", action="store_true", help="store whole frames + teacher output (test sets)")
    ap.add_argument("--weights", default=None, help="yolov8n-pose.pt (default: weights folder)")
    ap.add_argument("--out", default="outputs/p6/pose_train.npz")
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(2)
    from cinescope.pose.yolo import YoloPose

    teacher = YoloPose(args.weights)
    crops, kpts, src, frames_out, people_out = [], [], [], [], []
    t0 = time.time()
    for clip in args.clips:
        path = resolve(clip)
        n_before = len(crops) + len(frames_out)
        with VideoReader(path) as vr:
            width = min(vr.info.width, args.max_width)
            vr.width = width if width != vr.info.width else None
            batch = []
            for f in vr.frames(sample_fps=args.fps):
                batch.append(f)
                if len(batch) == 8:
                    _process(teacher, batch, clip_name(path), args, crops, kpts, src, frames_out, people_out)
                    batch = []
            _process(teacher, batch, clip_name(path), args, crops, kpts, src, frames_out, people_out)
        print(f"{clip_name(path)}: {len(crops) + len(frames_out) - n_before} samples ({time.time() - t0:.0f} s)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.keep_frames:
        np.savez_compressed(out, frames=np.array(frames_out, dtype=object),
                            people=np.array(people_out, dtype=object))
        print(f"{len(frames_out)} frames, {sum(len(p['boxes']) for p in people_out)} people -> {out}")
    else:
        np.savez_compressed(out, crops=np.stack(crops), kpts=np.stack(kpts), src=np.array(src))
        print(f"{len(crops)} crops -> {out}")


def _process(teacher, batch, name, args, crops, kpts, src, frames_out, people_out):
    if not batch:
        return
    people = teacher([f.rgb for f in batch])
    import cv2

    for f, p in zip(batch, people, strict=True):
        if args.keep_frames:
            frames_out.append({"clip": name, "t": f.t, "rgb": f.rgb})
            people_out.append({"boxes": p.boxes, "scores": p.scores, "kpts": p.kpts})
            continue
        for b, k in zip(p.boxes, p.kpts, strict=True):
            if b[3] - b[1] < args.min_height or (k[:, 2] >= 0.5).sum() < 6:
                continue
            c, s = box_to_center_scale(b, STORE_HW[1] / STORE_HW[0])
            M = crop_matrix(c, s * ENLARGE, STORE_HW)
            crop = cv2.warpAffine(f.rgb, M, (STORE_HW[1], STORE_HW[0]), flags=cv2.INTER_LINEAR)
            kk = k.copy()
            kk[:, :2] = transform_points(k[:, :2], M)
            crops.append(crop)
            kpts.append(kk.astype(np.float32))
            src.append(f"{name}@{f.t:.2f}")


if __name__ == "__main__":
    main()
