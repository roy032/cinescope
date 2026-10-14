"""Train the box-prompted person U-Net on a CPU, with no hand-made masks.

    python p4_segmentation/make_people.py --role train trailers/<training clips> ...
    python p4_segmentation/make_people.py --role trainbg --keep-empty trailers/<more clips> ...   # optional backgrounds
    python p4_segmentation/train_unet.py --steps 3000
    # -> p4_segmentation/checkpoints/unet_person.pt, results/unet_train.json

Every sample is a box crop (box + 15% margin, resized to 96 x 96) and a mask,
drawn half the time from each of two target sources:

  distilled   a real frame, a YOLOv8n-seg person box (jittered) and YOLOv8n-seg's
              mask as the target. Real boundaries and real occlusions, but the
              target is only as good as YOLOv8n-seg (coarse, 160x160 prototypes).
  composite   a YOLOv8n-seg cut-out from one training frame pasted at a random
              scale and place onto another training frame. The target is the
              pasted mask, exact by construction; the seams are unnaturally
              sharp, which is the price.

Boxes are jittered (each edge by N(0, 4% of the box size)) so the model does
not learn to trust the prompt to the pixel. The dev split (scored after
training, used to pick the step count) is composites + distilled crops from the
same training clips with a fixed seed; the test clips are never touched here.
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
import torch  # noqa: E402

from cinescope.segment.data import (  # noqa: E402
    composite,
    crop_padded,
    expand_box,
    load_people,
    make_cutout,
    mask_box,
    place,
)
from cinescope.segment.unet import CROP_MARGIN, CROP_SIZE, UNet, seg_loss  # noqa: E402


class Sampler:
    def __init__(self, frames, bgs, rng: np.random.Generator, p_distill: float = 0.5,
                 size: int = CROP_SIZE, margin: float = CROP_MARGIN, jitter: float = 0.04) -> None:
        self.inst = [(f, k) for f in frames for k in range(len(f.boxes))]
        self.cuts = [make_cutout(f.rgb, f.masks[k], f.clip) for f, k in self.inst
                     if f.boxes[k, 3] - f.boxes[k, 1] >= 64]
        self.bgs = bgs
        self.rng, self.p, self.size, self.margin, self.jitter = rng, p_distill, size, margin, jitter

    def _jitter(self, box):
        x0, y0, x1, y1 = box
        s = self.jitter * np.array([x1 - x0, y1 - y0, x1 - x0, y1 - y0])
        return np.asarray(box, np.float64) + self.rng.normal(0, 1, 4) * s

    def sample(self):
        rng = self.rng
        if rng.random() < self.p:
            f, k = self.inst[rng.integers(len(self.inst))]
            img, mask, box = f.rgb, f.masks[k], f.boxes[k]
        else:
            bg = self.bgs[rng.integers(len(self.bgs))]
            cut, xy = place(self.cuts[rng.integers(len(self.cuts))], bg.shape, rng)
            img, mask = composite(bg, cut, xy)
            box = mask_box(mask)
        box = self._jitter(box)
        h, w = img.shape[:2]
        bounds = expand_box(box, self.margin, h, w)
        if bounds[2] - bounds[0] < 4 or bounds[3] - bounds[1] < 4:
            return self.sample()
        c = cv2.resize(crop_padded(img, bounds), (self.size, self.size), interpolation=cv2.INTER_AREA)
        m = cv2.resize(crop_padded(mask.astype(np.float32), bounds), (self.size, self.size),
                       interpolation=cv2.INTER_AREA)
        return c, m

    def batch(self, n: int, augment: bool = True):
        xs, ys = zip(*(self.sample() for _ in range(n)), strict=True)
        x = np.stack(xs).astype(np.float32) / 255.0
        y = np.stack(ys)[:, None].astype(np.float32)
        if augment:
            flip = self.rng.random(n) < 0.5
            x[flip] = x[flip][:, :, ::-1]
            y[flip] = y[flip][:, :, ::-1]
            gain = self.rng.uniform(0.7, 1.3, (n, 1, 1, 1)).astype(np.float32)
            bias = self.rng.uniform(-0.1, 0.1, (n, 1, 1, 1)).astype(np.float32)
            x = np.clip(x * gain + bias, 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2),
                torch.from_numpy(np.ascontiguousarray(y)))


def evaluate(model, batches) -> float:
    model.eval()
    ious = []
    with torch.inference_mode():
        for x, y in batches:
            p = torch.sigmoid(model(x)) > 0.5
            t = y > 0.5
            inter = (p & t).sum((1, 2, 3)).float()
            union = (p | t).sum((1, 2, 3)).float().clamp(min=1)
            ious += (inter / union).tolist()
    model.train()
    return float(np.mean(ious))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="outputs/p4data")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--p-distill", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="p4_segmentation/checkpoints/unet_person.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    data = Path(args.data)
    frames = load_people(data / "people_train.npz")
    bgs = [f.rgb for f in frames]
    if (data / "people_trainbg.npz").exists():
        bgs += [f.rgb for f in load_people(data / "people_trainbg.npz")]
    # dev split: a fixed, separate stream from the same clips (scored, never trained on)
    dev_s = Sampler(frames, bgs, np.random.default_rng(10_000), args.p_distill)
    dev = [dev_s.batch(32, augment=False) for _ in range(8)]
    tr = Sampler(frames, bgs, np.random.default_rng(args.seed), args.p_distill)
    print(f"{len(tr.inst)} distilled instances, {len(tr.cuts)} cut-outs, {len(bgs)} backgrounds")

    model = UNet(base=args.base)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    log = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        torch.save({"model": model.state_dict(),
                    "config": {"base": args.base, "size": CROP_SIZE, "margin": CROP_MARGIN},
                    "args": vars(args), "log": log}, out)

    t0 = time.time()
    run = 0.0
    for step in range(1, args.steps + 1):
        x, y = tr.batch(args.bs)
        loss = seg_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            iou = evaluate(model, dev)
            log.append({"step": step, "loss": round(run, 4), "dev_iou": round(iou, 4),
                        "minutes": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
            save()   # checkpoint as we go: a CPU run can be interrupted
    res = Path("p4_segmentation/results/unet_train.json")
    res.write_text(json.dumps({"args": vars(args), "n_distilled": len(tr.inst), "n_cutouts": len(tr.cuts),
                               "n_backgrounds": len(bgs), "log": log}, indent=1))
    print("->", out, res)


if __name__ == "__main__":
    main()
