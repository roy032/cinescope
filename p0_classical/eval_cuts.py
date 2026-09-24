"""Score shot-boundary detection against your hand labels.

    python p0_classical/eval_cuts.py trailers/Ted_Lasso_900s.mkv p0_classical/labels/Ted_Lasso_900s.labels.json
    python p0_classical/eval_cuts.py --all            # every clip that has a label file

Compares, on the same frames:
  ours            histogram spikes + flash guard + fades/dissolves (cinescope.shots.cuts)
  ours-cuts-only  the same without the gradual-transition detectors
  pyscenedetect   ContentDetector and AdaptiveDetector, if `pip install scenedetect` is present
  ffmpeg-scene    FFmpeg's `select='gt(scene,0.3)'` filter, if ffmpeg is on PATH

Writes p0_classical/results/<clip>.json and prints a table. The detector
parameters are fixed in code and were chosen on the *_120s development clips;
the labelled *_900s clips are the held-out test set.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cinescope.io.video import VideoReader  # noqa: E402
from cinescope.shots import cuts, features  # noqa: E402
from cinescope.shots.evaluate import load_labels, score  # noqa: E402

LABELS = ROOT / "p0_classical" / "labels"
RESULTS = ROOT / "p0_classical" / "results"


def run_ours(video: str) -> dict[str, tuple[list[int], float]]:
    t0 = time.perf_counter()
    with VideoReader(video, width=160) as vr:
        feat = features.extract(vr.frames(), keep_thumbs=False)
    t_feat = time.perf_counter() - t0
    out = {}
    for name, gradual in (("ours", True), ("ours-cuts-only", False)):
        t1 = time.perf_counter()
        tr = cuts.detect_transitions(feat, gradual=gradual)
        out[name] = ([t.frame for t in tr], t_feat + time.perf_counter() - t1)
    return out


def run_pyscenedetect(video: str, fps: float) -> dict[str, tuple[list[int], float]]:
    try:
        from scenedetect import AdaptiveDetector, ContentDetector, detect
    except ImportError:
        return {}
    out = {}
    for name, det in (("pyscenedetect-content", ContentDetector()),
                      ("pyscenedetect-adaptive", AdaptiveDetector())):
        t0 = time.perf_counter()
        scenes = detect(video, det)
        frames = [s[0].get_frames() for s in scenes[1:]]      # start frame of every shot but the first
        out[name] = (frames, time.perf_counter() - t0)
    return out


def run_ffmpeg(video: str, fps: float, threshold: float = 0.3) -> dict[str, tuple[list[int], float]]:
    if not shutil.which("ffmpeg"):
        return {}
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-an", "-sn",
         "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True)
    times = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr)]
    return {f"ffmpeg-scene>{threshold}": ([int(round(t * fps)) for t in times], time.perf_counter() - t0)}


def evaluate(video: str, label_path: Path, tol: int = 2, baselines: bool = True) -> dict:
    labels = load_labels(label_path)
    with VideoReader(video) as vr:
        fps = vr.info.fps
    runs = run_ours(video)
    if baselines:
        runs.update(run_pyscenedetect(video, fps))
        runs.update(run_ffmpeg(video, fps))
    report = {"video": Path(video).name, "labels": str(label_path.name), "tolerance_frames": tol,
              "n_labelled": len(labels["transitions"]),
              "n_cuts": sum(t["kind"] == "cut" for t in labels["transitions"]),
              "methods": {}}
    for name, (frames, seconds) in runs.items():
        s = score(frames, labels["transitions"], tol=tol)
        report["methods"][name] = {**s["overall"], "cut_recall": s["cut"]["recall"],
                                   "gradual_recall": s["gradual"]["recall"],
                                   "seconds": round(seconds, 1),
                                   "false_positive_frames": s["false_positives"][:50],
                                   "missed": s["missed"][:50]}
    return report


def print_table(report: dict) -> None:
    print(f"\n{report['video']}: {report['n_labelled']} labelled transitions "
          f"({report['n_cuts']} cuts), tolerance ±{report['tolerance_frames']} frames")
    print(f"{'method':28s} {'P':>6s} {'R':>6s} {'F1':>6s} {'cutR':>6s} {'gradR':>6s} {'sec':>6s}")
    for name, m in report["methods"].items():
        gr = "—" if m["gradual_recall"] is None else f"{m['gradual_recall']:.3f}"
        print(f"{name:28s} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} "
              f"{m['cut_recall'] or 0:6.3f} {gr:>6s} {m['seconds']:6.1f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("labels", nargs="?")
    ap.add_argument("--all", action="store_true", help="every p0_classical/labels/*.labels.json")
    ap.add_argument("--clips", default="trailers", help="folder holding the clips for --all")
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args(argv)

    pairs: list[tuple[str, Path]] = []
    if args.all:
        for lp in sorted(LABELS.glob("*.labels.json")):
            name = json.loads(lp.read_text(encoding="utf-8"))["video"]
            pairs.append((str(Path(args.clips) / name), lp))
    elif args.video and args.labels:
        pairs.append((args.video, Path(args.labels)))
    else:
        ap.error("give VIDEO LABELS, or --all")

    RESULTS.mkdir(parents=True, exist_ok=True)
    totals: dict[str, list[int]] = {}
    for video, lp in pairs:
        rep = evaluate(video, lp, tol=args.tol, baselines=not args.no_baselines)
        (RESULTS / f"{Path(video).stem}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print_table(rep)
        for name, m in rep["methods"].items():
            t = totals.setdefault(name, [0, 0, 0])
            t[0] += m["tp"]
            t[1] += m["fp"]
            t[2] += m["fn"]
    if len(pairs) > 1:
        print("\nPooled over all clips (micro-averaged):")
        for name, (tp, fp, fn) in totals.items():
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            print(f"  {name:28s} P {p:.3f}  R {r:.3f}  F1 {f1:.3f}")


if __name__ == "__main__":
    main()
