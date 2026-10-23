"""Score every shot-scale system against the hand-labelled test shots.

    python p1_cnn/eval_own.py
    python p1_cnn/eval_own.py --set public       # the public-clip set: results/public, labels/public_test

Reads p1_cnn/labels/own_test.labels.json (copied there from Downloads if it is
only there), p1_cnn/results/own/teacher.csv and student_<tag>.csv, and writes
p1_cnn/results/own/hand_eval.json, hand_eval.md and confusion_<tag>.png.

Systems: each automatic cue alone, the fused teacher the networks were trained
on, and the two ResNet-18s. A student that beats its teacher on hand labels
has learned something the noisy labels only hinted at; one that trails it has
mostly copied the noise.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cinescope.framing import SCALES  # noqa: E402
from cinescope.framing.metrics import metrics, plot_confusion  # noqa: E402

RES = ROOT / "p1_cnn" / "results" / "own"
LABELS = ROOT / "p1_cnn" / "labels" / "own_test.labels.json"


def find_labels() -> Path:
    if LABELS.exists():
        return LABELS
    dl = sorted((Path.home() / "Downloads").glob(LABELS.name.replace(".json", "*.json")),
                key=lambda p: p.stat().st_mtime)
    if not dl:
        sys.exit("no hand labels yet: label the shots in outputs/label_scales/index.html and press "
                 "'Download labels' (or run p1_cnn/label_scales.py first)")
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(dl[-1], LABELS)
    print(f"copied {dl[-1]} -> {LABELS}")
    return LABELS


def bootstrap_ci(correct: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, len(correct), (n, len(correct)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def read_csv(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        return {f"{r['episode']}/{r['shot']}": r for r in csv.DictReader(f)}


def score(name: str, keys: list[str], y: np.ndarray, probs: dict[str, np.ndarray]) -> dict | None:
    idx = [i for i, k in enumerate(keys) if k in probs]
    if not idx:
        return None
    p = np.stack([probs[keys[i]] for i in idx])
    yy = y[idx]
    m = metrics(p, yy, SCALES)
    m["accuracy_ci95"] = bootstrap_ci((p.argmax(1) == yy).astype(float))
    m["coverage"] = len(idx) / len(keys)
    m["name"] = name
    return m


def _rel(p) -> str:
    """p relative to the repo, or absolute when it is on another drive (Windows)."""
    try:
        return os.path.relpath(p, ROOT).replace(os.sep, "/")
    except ValueError:
        return Path(p).resolve().as_posix()


def main(argv: list[str] | None = None) -> None:
    global RES, LABELS
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", help="shot set: results/<set>/, labels/<set>_test.labels.json (default: own)")
    args = ap.parse_args(argv)
    if args.set:
        RES = ROOT / "p1_cnn" / "results" / args.set
        LABELS = ROOT / "p1_cnn" / "labels" / f"{args.set}_test.labels.json"
    lab = json.loads(find_labels().read_text(encoding="utf-8"))["labels"]
    lab = {k: v for k, v in lab.items() if v in SCALES}
    if len(lab) < 30:
        sys.exit(f"only {len(lab)} labelled shots: label more before evaluating")
    keys = sorted(lab)
    y = np.array([SCALES.index(lab[k]) for k in keys])
    teacher = read_csv(RES / "teacher.csv")

    def cols(prefix: str) -> dict[str, np.ndarray]:
        out = {}
        for k, r in teacher.items():
            if r[f"{prefix}_{SCALES[0]}"] != "":
                out[k] = np.array([float(r[f"{prefix}_{c}"]) for c in SCALES])
        return out

    meta_path = RES / "teacher.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    text_cue = meta.get("text_cue", "SigLIP").split(" (")[0]
    face_cue = meta.get("face_cue", "")
    systems = [(f"{text_cue} zero-shot (prompts only)", "text", cols("text")),
               (f"Face size only{f' ({face_cue})' if face_cue else ''}, shots with a face", "face", cols("face")),
               ("Teacher: prompts + face size", "fused", cols("fused"))]
    for tag, desc in (("pretrained", "ResNet-18, ImageNet init"), ("scratch", "ResNet-18, random init")):
        f = RES / f"student_{tag}.csv"
        if f.exists():
            systems.append((desc, tag, {k: np.array([float(r[f"p_{c}"]) for c in SCALES])
                                        for k, r in read_csv(f).items()}))

    results = {}
    for name, key, probs in systems:
        m = score(name, keys, y, probs)
        if m is not None:
            results[key] = m
            if key in ("pretrained", "scratch", "fused"):
                plot_confusion(np.array(m["confusion"]), SCALES, str(RES / f"confusion_{key}.png"),
                               f"{name}: acc {m['accuracy']:.3f} (n={m['n']})")

    dist = {c: int((y == i).sum()) for i, c in enumerate(SCALES)}
    lines = [f"Hand-labelled test shots: {len(keys)} from the held-out videos ({dist}); "
             f"majority-class accuracy {max(dist.values()) / len(keys):.3f}.", "",
             "| System | Accuracy [95% CI] | Mean class acc. | Shots scored |", "|---|---|---|---|"]
    for m in results.values():
        lo, hi = m["accuracy_ci95"]
        lines.append(f"| {m['name']} | {m['accuracy']:.3f} [{lo:.2f}–{hi:.2f}] | {m['mean_class_accuracy']:.3f} | "
                     f"{m['n']} ({m['coverage']:.0%}) |")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "hand_eval.json").write_text(json.dumps({"n": len(keys), "distribution": dist, "teacher": meta,
                                                    "labels": _rel(LABELS),
                                                    "systems": results},
                                                   indent=1), encoding="utf-8")
    (RES / "hand_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
