"""Train the CenterNet face detector on WIDER FACE (Phase 2, week 7).

    # GPU (Kaggle / Colab), ImageNet-initialised backbone
    python p2_detection/train_faces.py --wider /kaggle/input/wider-face --epochs 70 --bs 32

    # 1-minute CPU smoke test, no dataset
    python p2_detection/train_faces.py --synthetic --epochs 1

Expected layout under --wider (the official downloads, unzipped):
    WIDER_train/images/...  WIDER_val/images/...  wider_face_split/*.txt
Writes p2_detection/results/<tag>/history.csv and checkpoints/<tag>.pt.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.detect.centernet import CenterNet, centernet_loss  # noqa: E402
from cinescope.detect.centernet_targets import encode  # noqa: E402
from cinescope.detect.wider import WiderTrain, parse_annotations  # noqa: E402
from cinescope.framing.train import param_groups, warmup_cosine  # noqa: E402

HERE = Path(__file__).resolve().parent


class SyntheticFaces:
    """Bright ellipses ("faces") of random size on noise, with exact boxes."""

    def __init__(self, n: int, size: int = 128, seed: int = 0):
        self.n, self.size, self.seed = n, size, seed

    def __len__(self) -> int:
        return self.n

    def sample(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng((self.seed, i))
        s = self.size
        img = rng.normal(0, 0.3, (3, s, s)).astype(np.float32)
        yy, xx = np.mgrid[:s, :s] + 0.5
        boxes = []
        for _ in range(int(rng.integers(1, 5))):
            w = rng.uniform(10, 40)
            h = w * rng.uniform(1.1, 1.4)
            cx, cy = rng.uniform(w / 2, s - w / 2), rng.uniform(h / 2, s - h / 2)
            img[:, ((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2 < 1] = 2.0
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        return img, np.array(boxes)

    def __getitem__(self, i: int) -> dict:
        img, boxes = self.sample(i)
        t = encode(boxes, np.zeros(len(boxes), np.int64), (self.size // 4,) * 2)
        return {"image": torch.from_numpy(img), **{k: torch.from_numpy(v) for k, v in t.items()}}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wider", help="WIDER FACE root")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="peak AdamW LR (default 5e-4 x bs/32)")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--no-pretrained", action="store_true", help="random-init backbone")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--tag", default="centernet_r18")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--ckpt-dir", default=str(HERE / "checkpoints"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not args.synthetic and not args.wider:
        ap.error("--wider is required unless --synthetic")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt_dir) / f"{args.tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        ds = SyntheticFaces(64)
        pretrained = False
    else:
        root = Path(args.wider)
        items = [it for it in parse_annotations(root / "wider_face_split" / "wider_face_train_bbx_gt.txt")
                 if len(it.boxes)]
        ds = WiderTrain(root / "WIDER_train" / "images", items, size=args.size)
        pretrained = not args.no_pretrained
    workers = 0 if args.synthetic else args.workers
    dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=workers,
                    pin_memory=use_amp, persistent_workers=workers > 0)

    model = CenterNet(pretrained_backbone=pretrained).to(device).to(memory_format=torch.channels_last)
    lr = args.lr or 5e-4 * args.bs / 32
    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=lr)
    steps = min(len(dl), args.max_steps or len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(args.epochs * steps, min(500, steps)))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"{device} amp={use_amp} | {len(ds)} images | {steps} steps/epoch | lr {lr:.2e} | "
          f"backbone {'ImageNet' if pretrained else 'random'}")

    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, sums, n = time.perf_counter(), {}, 0
        for step, batch in enumerate(dl):
            if step >= steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            x = batch["image"].contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, enabled=use_amp):
                outputs = model(x)
            losses = centernet_loss(outputs, batch)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"loss is {losses['loss'].item()} at epoch {ep} step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 35.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n += 1
        row = {"epoch": ep, **{k: round(v / max(n, 1), 4) for k, v in sums.items()},
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.perf_counter() - t0, 1)}
        history.append(row)
        print(" | ".join(f"{k} {v}" for k, v in row.items()))
        torch.save({"model": model.state_dict(), "n_classes": 1, "epoch": ep, "args": vars(args)}, ckpt)

    with (out / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    print(f"checkpoint {ckpt}; evaluate with p2_detection/eval_wider.py --ckpt {ckpt}")
    return {"history": history, "checkpoint": str(ckpt)}


if __name__ == "__main__":
    main()
