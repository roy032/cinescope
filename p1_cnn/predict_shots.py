"""Class probabilities of a trained framing model for every shot of one split.

    python p1_cnn/predict_shots.py --ckpt p1_cnn/checkpoints/pretrained.pt --root /tmp/own \\
        --shots /tmp/own/shots.csv --split test --out p1_cnn/results/own/student_pretrained.csv

Each shot is scored on its three key frames and the probabilities are averaged,
as in evaluation during training.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> None:
    import torch
    from torch.utils.data import DataLoader

    from cinescope.framing.data import ShotDataset, ShotRecord
    from cinescope.framing.resnet import resnet18

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--shots", required=True, help="shots.csv from build_own_shots.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=dev, weights_only=False)
    classes = tuple(state["classes"])
    model = resnet18(len(classes))
    model.load_state_dict(state["model"])
    model.to(dev).eval()

    with open(args.shots, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == args.split]
    recs = [ShotRecord(r["split"], r["episode"], r["shot"], classes[0], "", r["frames"].split(";")) for r in rows]
    ds = ShotDataset(recs, args.root, train=False, size=tuple(state.get("size", (224, 384))), classes=classes)
    probs = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, args.bs, num_workers=2):
            b, v = x.shape[:2]
            p = model(x.flatten(0, 1).to(dev)).softmax(1).view(b, v, -1).mean(1)
            probs.append(p.cpu().numpy())
    probs = np.concatenate(probs) if probs else np.zeros((0, len(classes)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "shot", "label", *[f"p_{c}" for c in classes]])
        for r, p in zip(rows, probs, strict=True):
            w.writerow([r["episode"], r["shot"], classes[int(p.argmax())], *[round(float(v), 4) for v in p]])
    print(f"{len(rows)} {args.split} shots -> {args.out}")


if __name__ == "__main__":
    main()
