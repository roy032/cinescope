"""The whole Phase 1 experiment on public clips, on a CPU: teacher, two students, hand-label scores.

    python p1_cnn/build_own_shots.py --videos data/public_clips --out data/public_shots --no-zip
    python p1_cnn/run_public.py                       # ~1 h on 2 CPU threads
    python tools/update_results.py

A few short public clips give ~70 shots, too few for the usual held-out 20%.
So the clips are split by time into two halves and everything runs twice
(2-fold cross-validation by time): fold 0 trains on the second half of every
clip and predicts the first, fold 1 the reverse (the 10% of the clip just
before the test half, wrapping round to the end for fold 0, is validation). Every shot is predicted exactly once by
a network that never saw it, and the hand labels cover all of them.

Steps
  1. shots_fold{0,1}.csv with the fold splits (build_own_shots.assign_splits)
  2. the teacher on every shot (autolabel.py: open_clip zero-shot + YuNet face size,
     the CPU back-ends) -> results/public/teacher.csv
  3. per fold, ResNet-18 from ImageNet weights and from random init on the teacher's
     labels of that fold's train/val shots -> results/public_f<k>_<init>/
  4. predictions for each fold's test half, merged -> results/public/student_<init>.csv;
     a summary of both folds' training -> results/public_<init>/
  5. eval_own.py --set public (every system against p1_cnn/labels/public_test.labels.json)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "p1_cnn"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

INITS = ("pretrained", "scratch")


def weights_dir() -> Path:
    return Path(os.environ.get("CINESCOPE_WEIGHTS", ROOT / "data" / "weights"))


def write_folds(shots_root: Path, folds: int, val_frac: float) -> list[Path]:
    from build_own_shots import assign_splits

    eps = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((shots_root / "episodes").glob("*.json"))]
    out = []
    for k in range(folds):
        split = assign_splits(eps, val_frac=val_frac, fold=k, folds=folds)
        path = shots_root / f"shots_fold{k}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "episode", "shot", "start", "end", "frames"])
            for e in eps:
                for s in e["shots"]:
                    w.writerow([split[(e["episode"], s["shot"])], e["episode"], s["shot"], s["start"], s["end"],
                                ";".join(s["frames"])])
        n = {r: sum(v == r for v in split.values()) for r in ("train", "val", "test")}
        print(f"fold {k}: {n}")
        out.append(path)
    return out


def fold_index(teacher_csv: Path, shots_csv: Path, out: Path) -> None:
    """train/val rows with the teacher's label, for one fold's splits."""
    with teacher_csv.open(encoding="utf-8") as f:
        label = {(r["episode"], r["shot"]): r["label"] for r in csv.DictReader(f)}
    with shots_csv.open(encoding="utf-8") as f, out.open("w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["split", "trailer", "shot", "scale", "movement", "frames"])
        for r in csv.DictReader(f):
            if r["split"] in ("train", "val"):
                w.writerow([r["split"], r["episode"], r["shot"], label[(r["episode"], r["shot"])], "", r["frames"]])


def summarise(res: Path, init: str, folds: int) -> None:
    """results/public_<init>/: both folds' histories (one after the other) and a metrics.json."""
    runs = [res / f"public_f{k}_{init}" for k in range(folds)]
    ms = [json.loads((r / "metrics.json").read_text(encoding="utf-8")) for r in runs]
    out = res / f"public_{init}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, r in enumerate(runs):
        with (r / "history.csv").open(encoding="utf-8") as f:
            rows += [{"fold": k, **row} for row in csv.DictReader(f)]
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    m = {k: ms[0][k] for k in ("tag", "pretrained", "epochs", "lr", "batch_size", "size", "weights", "device")}
    m.update(tag=f"public_{init}", folds=folds, best_epoch_per_fold=[x["best_epoch"] for x in ms],
             best_val_acc_vs_teacher_per_fold=[round(x["best_val_acc"], 4) for x in ms])
    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots-root", default=str(ROOT / "data" / "public_shots"))
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--imagenet", default=str(weights_dir() / "resnet18_a1_0-d63eafa0.pth"),
                    help="ImageNet ResNet-18 state dict (torchvision names); 'torchvision' to download")
    ap.add_argument("--epochs-pretrained", type=int, default=30)
    ap.add_argument("--epochs-scratch", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    # The recipe's lr (0.01 / 0.1 per 256 images) assumes thousands of steps; at batch 8 on ~30 shots
    # it would barely move the weights in 30-60 epochs. These were fixed before any run, not tuned.
    ap.add_argument("--lr-pretrained", type=float, default=0.01)
    ap.add_argument("--lr-scratch", type=float, default=0.05)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--skip-teacher", action="store_true", help="reuse results/public/teacher.csv")
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    import autolabel
    import eval_own
    import predict_shots
    import train_framing

    sr, res = Path(args.shots_root), HERE / "results"
    pub = res / "public"
    pub.mkdir(parents=True, exist_ok=True)
    fold_csvs = write_folds(sr, args.folds, args.val_frac)
    t0 = time.time()
    if not args.skip_teacher:
        autolabel.main(["--root", str(sr), "--shots", str(fold_csvs[0]), "--text", "openclip", "--out", str(pub)])
        print(f"teacher: {(time.time() - t0) / 60:.1f} min", flush=True)

    ck = ROOT / "outputs" / "p1_public_checkpoints"
    for init in INITS:
        merged = []
        for k, shots in enumerate(fold_csvs):
            index = pub / f"index_fold{k}.csv"
            fold_index(pub / "teacher.csv", shots, index)
            tag = f"public_f{k}_{init}"
            cmd = ["--root", str(sr), "--index", str(index), "--out", str(res), "--ckpt-dir", str(ck),
                   "--tag", tag, "--bs", str(args.bs), "--workers", "0", "--height", str(args.height),
                   "--width", str(args.width), "--warmup", "2"]
            if init == "pretrained":
                cmd += ["--pretrained", "--epochs", str(args.epochs_pretrained), "--lr", str(args.lr_pretrained)]
                if args.imagenet != "torchvision":
                    cmd += ["--weights", args.imagenet]
            else:
                cmd += ["--epochs", str(args.epochs_scratch), "--lr", str(args.lr_scratch)]
            train_framing.main(cmd)
            pred = pub / f"student_{tag}.csv"
            predict_shots.main(["--ckpt", str(ck / f"{tag}.pt"), "--root", str(sr), "--shots", str(shots),
                                "--split", "test", "--out", str(pred)])
            with pred.open(encoding="utf-8") as f:
                merged += list(csv.DictReader(f))
        with (pub / f"student_{init}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(merged[0]))
            w.writeheader()
            w.writerows(merged)
        summarise(res, init, args.folds)
    print(f"all done in {(time.time() - t0) / 60:.1f} min", flush=True)
    eval_own.main(["--set", "public"])


if __name__ == "__main__":
    main()
