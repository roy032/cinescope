"""Train the ResNet-18 shot-scale classifier (Phase 1, week 5).

    # the comparison the roadmap asks for (GPU: Kaggle / Colab)
    python p1_cnn/train_framing.py --root D:/datasets/MovieShots --pretrained --tag pretrained
    python p1_cnn/train_framing.py --root D:/datasets/MovieShots --tag scratch --epochs 60

    # 1-minute CPU smoke test of the whole loop, no dataset needed
    python p1_cnn/train_framing.py --synthetic --epochs 2 --tag smoke

Writes p1_cnn/results/<tag>/: metrics.json (val per epoch, test at the best
val epoch), confusion.png, history.csv; checkpoints/<tag>.pt (git-ignored).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.framing import SCALES  # noqa: E402
from cinescope.framing.data import ShotDataset, SyntheticShots, read_index  # noqa: E402
from cinescope.framing.metrics import metrics, plot_confusion  # noqa: E402
from cinescope.framing.resnet import count_parameters, resnet18  # noqa: E402
from cinescope.framing.train import (  # noqa: E402
    param_groups,
    predict,
    train_one_epoch,
    warmup_cosine,
)

HERE = Path(__file__).resolve().parent


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_data(args):
    if args.synthetic:
        return (SyntheticShots(args.synthetic_n, True, seed=0), SyntheticShots(args.synthetic_n // 4, False, seed=1),
                SyntheticShots(args.synthetic_n // 4, False, seed=2))
    index = Path(args.index)
    if not index.exists():
        sys.exit(f"{index} not found — run p1_cnn/prepare_movieshots.py --root {args.root} first")
    size = (args.height, args.width)
    splits = [read_index(index, s) for s in ("train", "val", "test")]
    if not splits[0] or not splits[1]:
        sys.exit(f"{index} has no train/val rows")
    return (ShotDataset(splits[0], args.root, True, size), ShotDataset(splits[1], args.root, False, size),
            ShotDataset(splits[2], args.root, False, size))


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="MovieShots root (images are read relative to it)")
    ap.add_argument("--index", default="data/movieshots/index.csv")
    ap.add_argument("--synthetic", action="store_true", help="smoke-test on generated images")
    ap.add_argument("--synthetic-n", type=int, default=800)
    ap.add_argument("--pretrained", action="store_true", help="start from ImageNet weights")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, help="peak LR (default 0.1 scratch / 0.01 pretrained, x bs/256)")
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=float, default=1.0, help="warm-up epochs")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--max-steps", type=int, help="cap steps per epoch (debugging)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", help="results folder (default p*/results; outputs/smoke for --synthetic)")
    ap.add_argument("--ckpt-dir", help="default p*/checkpoints; outputs/smoke for --synthetic")
    args = ap.parse_args(argv)
    if not args.synthetic and not args.root:
        ap.error("--root is required unless --synthetic")

    seed_everything(args.seed)
    tag = args.tag or ("pretrained" if args.pretrained else "scratch")
    # smoke runs must not leave files where real results go
    smoke = Path("outputs") / "smoke"
    args.out = args.out or str(smoke if args.synthetic else HERE / "results")
    args.ckpt_dir = args.ckpt_dir or str(smoke if args.synthetic else HERE / "checkpoints")
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt_dir) / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    train_ds, val_ds, test_ds = build_data(args)
    kw = {"num_workers": 0 if args.synthetic else args.workers, "pin_memory": device.type == "cuda",
          "persistent_workers": not args.synthetic and args.workers > 0}
    train_dl = DataLoader(train_ds, args.bs, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, max(1, args.bs // 3), **kw)
    test_dl = DataLoader(test_ds, max(1, args.bs // 3), **kw) if len(test_ds) else None

    model = resnet18(len(SCALES), pretrained=args.pretrained).to(device).to(memory_format=torch.channels_last)
    lr = args.lr or (0.01 if args.pretrained else 0.1) * args.bs / 256
    opt = torch.optim.SGD(param_groups(model, args.wd), lr=lr, momentum=0.9, nesterov=True)
    steps_per_epoch = min(len(train_dl), args.max_steps or len(train_dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(args.epochs * steps_per_epoch, int(args.warmup * steps_per_epoch)))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    counts = np.bincount(train_ds.labels(), minlength=len(SCALES))
    print(f"{device} amp={use_amp} | ResNet-18 {count_parameters(model) / 1e6:.2f}M params | "
          f"{'ImageNet init' if args.pretrained else 'random init'} | lr {lr:.4f} | "
          f"train {len(train_ds)} val {len(val_ds)} test {len(test_ds)} | "
          f"train classes {dict(zip(SCALES, counts.tolist(), strict=True))}")

    history, best = [], (-1.0, -1)
    for ep in range(1, args.epochs + 1):
        st = train_one_epoch(model, train_dl, opt, sched, device, scaler, args.label_smoothing, args.max_steps)
        vp, vy = predict(model, val_dl, device, args.max_steps)
        vm = metrics(vp, vy, SCALES)
        history.append({"epoch": ep, "train_loss": round(st.loss, 4), "train_acc": round(st.acc, 4),
                        "val_acc": round(vm["accuracy"], 4), "val_mca": round(vm["mean_class_accuracy"], 4),
                        "lr": st.lr, "seconds": round(st.seconds, 1)})
        print(f"ep {ep:3d} | loss {st.loss:.3f} train {st.acc:.3f} | val acc {vm['accuracy']:.3f} "
              f"mca {vm['mean_class_accuracy']:.3f} | {st.seconds:.0f}s")
        if vm["accuracy"] > best[0]:
            best = (vm["accuracy"], ep)
            torch.save({"model": model.state_dict(), "classes": SCALES, "epoch": ep,
                        "size": (args.height, args.width), "args": vars(args)}, ckpt)

    with (out / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    result = {"tag": tag, "pretrained": args.pretrained, "best_epoch": best[1], "best_val_acc": best[0],
              "epochs": args.epochs, "lr": lr, "batch_size": args.bs, "synthetic": args.synthetic,
              "finished": time.strftime("%Y-%m-%d %H:%M")}
    if test_dl is not None:
        tp, ty = predict(model, test_dl, device, args.max_steps)
        tm = metrics(tp, ty, SCALES)
        result["test"] = tm
        plot_confusion(np.array(tm["confusion"]), SCALES, str(out / "confusion.png"),
                       f"{tag}: test acc {tm['accuracy']:.3f}")
        print(f"test acc {tm['accuracy']:.3f} | mean class acc {tm['mean_class_accuracy']:.3f}")
        print("most confused (true -> predicted):",
              ", ".join(f"{c['true']}->{c['pred']} {c['count']}" for c in tm["top_confusions"]))
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"results in {out}")
    return result


if __name__ == "__main__":
    main()
