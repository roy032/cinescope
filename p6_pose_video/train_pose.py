"""Train the SimpleBaseline pose network on teacher-labelled crops (CPU).

    python p6_pose_video/train_pose.py --data outputs/p6/pose_train.npz --minutes 25 --out outputs/p6/pose_sb.pt

Input crops 128 x 96, heatmaps 32 x 24 (sigma 1.5 cells), ResNet-18 backbone
initialised from ImageNet (`--scratch` for random init), 3 deconvs of 128
channels (the paper uses 256; half is plenty for 32 x 24 maps and twice as
fast on a CPU). Augmentation: scale 0.75-1.25, rotation +-30 degrees (60% of
samples), horizontal flip with left/right keypoints swapped, brightness and
contrast jitter. Adam 1e-3, cosine decay over the time budget.

A keypoint counts as labelled when the teacher's confidence is >= 0.5; others
get zero loss weight. So the network learns the teacher's confident outputs,
and agreement with the teacher is the most it can be expected to reach.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.pose import FLIP_PAIRS  # noqa: E402
from cinescope.pose.heatmaps import crop_to_heatmap, gaussian_targets  # noqa: E402
from cinescope.pose.simplebaseline import SimpleBaseline, flip_heatmaps, heatmap_loss, to_tensor  # noqa: E402

IN_HW, HM_HW, SIGMA = (128, 96), (32, 24), 1.5
ENLARGE = 1.3


def augment(crop: np.ndarray, kp: np.ndarray, rng: np.random.Generator, train: bool = True
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stored crop + keypoints -> network input, heatmap targets, weights."""
    sh, sw = crop.shape[:2]
    c = np.array([(sw - 1) / 2, (sh - 1) / 2])
    size = np.array([sw, sh]) / ENLARGE                 # the nominal (padded-box) window
    rot, flip = 0.0, False
    if train:
        size = size * rng.uniform(0.75, 1.25)
        c = c + rng.normal(0, 0.05, 2) * size
        if rng.random() < 0.6:
            rot = rng.uniform(-30, 30)
        flip = rng.random() < 0.5
    oh, ow = IN_HW
    s = (ow - 1) / size[0]
    R = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), rot, 1.0)          # rotate about the centre
    A = np.vstack([R, [0, 0, 1]])
    S = np.array([[s, 0, (ow - 1) / 2 - s * c[0]], [0, s, (oh - 1) / 2 - s * c[1]], [0, 0, 1]])
    M = (S @ A)[:2]
    img = cv2.warpAffine(crop, M, (ow, oh), flags=cv2.INTER_LINEAR)
    pts = kp[:, :2] @ M[:, :2].T + M[:, 2]
    vis = (kp[:, 2] >= 0.5).astype(np.float32)
    if flip:
        img = img[:, ::-1]
        pts[:, 0] = ow - 1 - pts[:, 0]
        idx = list(range(len(pts)))
        for a, b in FLIP_PAIRS:
            idx[a], idx[b] = b, a
        pts, vis = pts[idx], vis[idx]
    if train:
        img = np.clip(img.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
    hm, w = gaussian_targets(crop_to_heatmap(pts, IN_HW, HM_HW), vis, HM_HW, SIGMA)
    return np.ascontiguousarray(img), hm, w


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="outputs/p6/pose_train.npz")
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--deconv", type=int, default=128)
    ap.add_argument("--scratch", action="store_true", help="random init instead of ImageNet")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="outputs/p6/pose_sb.pt")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    d = np.load(args.data)
    crops, kpts = d["crops"], d["kpts"]
    n = len(crops)
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[: max(32, n // 20)], perm[max(32, n // 20):]
    print(f"{n} crops: {len(tr_idx)} train / {len(val_idx)} val")

    arch = {"deconv": args.deconv}
    net = SimpleBaseline(**arch)
    if not args.scratch:
        print(f"ImageNet init: {len(net.load_imagenet())} tensors")
    opt = torch.optim.Adam(net.parameters(), args.lr)
    budget = args.minutes * 60
    t0, step, log = time.time(), 0, []

    def val_loss():
        net.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(val_idx), 64):
                b = [augment(crops[j], kpts[j], rng, train=False) for j in val_idx[i:i + 64]]
                x = to_tensor(np.stack([e[0] for e in b]))
                hm = torch.tensor(np.stack([e[1] for e in b]))
                w = torch.tensor(np.stack([e[2] for e in b]))
                p = 0.5 * (net(x) + flip_heatmaps(net(x.flip(-1))))
                tot += float(heatmap_loss(p, hm, w)) * len(b)
        net.train()
        return tot / len(val_idx)

    net.train()
    while True:
        el = time.time() - t0
        if el > budget:
            break
        lr = args.lr * 0.5 * (1 + np.cos(np.pi * min(1.0, el / budget)))
        for g in opt.param_groups:
            g["lr"] = lr
        b = [augment(crops[j], kpts[j], rng) for j in rng.choice(tr_idx, args.bs)]
        x = to_tensor(np.stack([e[0] for e in b]))
        hm = torch.tensor(np.stack([e[1] for e in b]))
        w = torch.tensor(np.stack([e[2] for e in b]))
        loss = heatmap_loss(net(x), hm, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0:
            vl = val_loss()
            log.append({"step": step, "seconds": round(time.time() - t0), "train": loss.item(), "val": vl})
            print(f"step {step} {time.time() - t0:.0f}s lr {lr:.2e} loss {loss.item():.6f} val {vl:.6f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "arch": arch, "input_hw": IN_HW, "steps": step, "log": log,
                "imagenet": not args.scratch, "n_train": len(tr_idx)}, args.out)
    print(f"saved {args.out} after {step} steps")


if __name__ == "__main__":
    main()
