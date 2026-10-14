"""Cut clips into short windows, store small frame stacks and CLIP teacher scores.

    python p6_pose_video/build_scene_data.py --out outputs/p6/scenes_train.npz                 # training clips
    python p6_pose_video/build_scene_data.py --clips goldeneye Megamind --out outputs/p6/scenes_test.npz

Each Phase 0 shot is split into windows of about 2 s (cinescope.video.clips.
windows). Per window:

  frames  8 frames at 6 fps around the window centre (fewer seconds if the shot
          is shorter), centre-cropped to 7:4 and resized to 64 x 112: the input
          of the from-scratch models (a VideoMAE clip is 4 of them)
  logits  the OpenCLIP zero-shot class scores (cinescope/video/scenes.py) of 3
          frames at 25/50/75% of the window, averaged: the teacher's opinion

Test clips are stored the same way; their teacher scores are the library
baseline, and the hand labels (p6_pose_video/labels/*_scenes.json) score both.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.video.clips import clip_name, resolve, shots, windows  # noqa: E402
from cinescope.video.frames import frames_at  # noqa: E402

TRAIN_CLIPS = ["vtest", "classroom", "face-demographics-walking", "head-pose-face-detection-female-and-male",
               "one-by-one-person-detection", "people-detection", "person-bicycle-car-detection", "car-detection",
               "store-aisle-detection", "worker-zone-detection", "testvideo", "echo-hereweare", "big_buck_bunny"]
TEST_CLIPS = ["goldeneye", "Megamind"]
HW = (64, 112)
N_FRAMES, FPS = 8, 6.0


def small(rgb: np.ndarray, hw: tuple[int, int] = HW) -> np.ndarray:
    """Centre-crop to the target aspect ratio, then resize."""
    h, w = rgb.shape[:2]
    a = hw[1] / hw[0]
    if w / h > a:
        nw = int(round(h * a))
        rgb = rgb[:, (w - nw) // 2:(w - nw) // 2 + nw]
    else:
        nh = int(round(w / a))
        rgb = rgb[(h - nh) // 2:(h - nh) // 2 + nh]
    return cv2.resize(rgb, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def window_times(t0: float, t1: float) -> tuple[list[float], list[float]]:
    span = min((N_FRAMES - 1) / FPS, (t1 - t0) * 0.9)
    c = 0.5 * (t0 + t1)
    clip_t = list(np.linspace(c - span / 2, c + span / 2, N_FRAMES))
    clip_t_teacher = [t0 + f * (t1 - t0) for f in (0.25, 0.5, 0.75)]
    return clip_t, clip_t_teacher


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", default=TRAIN_CLIPS)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--out", default="outputs/p6/scenes_train.npz")
    args = ap.parse_args(argv)
    import torch

    torch.set_num_threads(2)
    teacher = None
    if not args.no_teacher:
        from cinescope.video.scenes import ClipTeacher

        teacher = ClipTeacher()
    rows = {"frames": [], "logits": [], "clip": [], "shot": [], "t0": [], "t1": []}
    t_start = time.time()
    for clip in args.clips:
        path = resolve(clip)
        sh = shots(path)["shots"]
        wins = windows(sh, args.seconds, min_seconds=0.0)
        times, spans = [], []
        for _, a, b in wins:
            ct, tt = window_times(a, b)
            spans.append((len(times), len(ct), len(tt)))
            times += ct + tt
        frames = frames_at(path, times, width=320)
        for (s, a, b), (i0, nc, nt) in zip(wins, spans, strict=True):
            fr = frames[i0:i0 + nc]
            rows["frames"].append(np.stack([small(f) for f in fr]))
            if teacher is not None:
                rows["logits"].append(teacher.frame_logits(frames[i0 + nc:i0 + nc + nt]).mean(0))
            rows["clip"].append(clip_name(path))
            rows["shot"].append(s)
            rows["t0"].append(a)
            rows["t1"].append(b)
        print(f"{clip_name(path)}: {len(sh)} shots, {len(wins)} windows ({time.time() - t_start:.0f} s)", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frames=np.stack(rows["frames"]),
                        logits=np.array(rows["logits"]) if rows["logits"] else np.zeros((len(rows["clip"]), 0)),
                        clip=np.array(rows["clip"]), shot=np.array(rows["shot"]),
                        t0=np.array(rows["t0"]), t1=np.array(rows["t1"]))
    print(f"{len(rows['clip'])} windows -> {out}")


if __name__ == "__main__":
    main()
