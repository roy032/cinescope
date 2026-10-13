"""Score face trackers: MOTA / IDF1 on labelled clips and on synthetic sequences (Week 9).

    python p3_tracking/eval_mot.py --clips trailers/goldeneye.mp4 trailers/head-pose-face-detection-female-and-male.mp4 trailers/face-demographics-walking.mp4
    python p3_tracking/eval_mot.py --synthetic 200

Clips: ground truth is p3_tracking/labels/<clip>.ids.json (identities of the
faces on every N-th frame; see labels/README.md). A ground-truth object is
"this identity within this shot" -- a tracker is not expected to follow a
person across a cut (that is clustering's job, Week 10). Only labelled frames
are scored. Every tracker gets the same YuNet detections.

Synthetic: sequences from cinescope/track/synthetic.py with exact ground truth
(misses, occlusions, jitter, low-score dips, false positives, cuts).

Results -> p3_tracking/results/mot.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cinescope.track.mot import evaluate, tracks_to_frames  # noqa: E402
from cinescope.track.tracker import Tracker, TrackerParams  # noqa: E402

TRACKERS = {
    "ours (ByteTrack-style, cuts reset)": lambda: Tracker(),
    "ours, no low-score round (SORT-like)": lambda: Tracker(TrackerParams(low=0.6)),
    "ours, no cut reset": "nocuts",
    "ultralytics ByteTrack": lambda: _ul("bytetrack"),
    "ultralytics ByteTrack, our thresholds": lambda: _ul("bytetrack", track_high_thresh=0.6, track_low_thresh=0.3,
                                                         new_track_thresh=0.7),
    "ultralytics BoT-SORT (no GMC, no ReID)": lambda: _ul("botsort", gmc_method=None),
}


def _ul(kind, **kw):
    from cinescope.track.ultra import UltralyticsTracker

    return UltralyticsTracker(kind, **kw)


def load_gt(path: Path, shot_starts: list[int]):
    lab = json.loads(path.read_text())
    starts = np.asarray(shot_starts)
    gt, ignore = {}, {}
    for fr in lab["frames"]:
        k = fr["index"]
        shot = int(np.searchsorted(starts, k, side="right") - 1)
        ids, boxes, ign = [], [], []
        for b in fr["boxes"]:
            if b["id"] == "ignore":
                ign.append(b["bbox"])
            else:
                ids.append(f"{b['id']}@{shot}")
                boxes.append(b["bbox"])
        gt[k] = (ids, np.asarray(boxes, dtype=np.float64).reshape(-1, 4))
        ignore[k] = np.asarray(ign, dtype=np.float64).reshape(-1, 4)
    return gt, ignore


def run_tracker(make, dets, cuts):
    trk = Tracker() if make == "nocuts" else make()
    cuts = set() if make == "nocuts" else set(cuts)
    t0 = time.perf_counter()
    for k, fd in enumerate(dets):
        if k in cuts and k > 0:
            trk.reset()
        b, s = (fd.boxes, fd.scores) if hasattr(fd, "boxes") else fd
        trk.update(k, b, s)
    tracks = trk.finish()
    return tracks, (time.perf_counter() - t0) / max(len(dets), 1) * 1000


def eval_clips(clips: list[str], out_dir: Path) -> dict:
    from p3_tracking.track_clip import load_or_cut, load_or_detect

    res = {}
    for video in clips:
        stem = Path(video).stem
        lab = ROOT / "p3_tracking" / "labels" / f"{stem}.ids.json"
        if not lab.exists():
            print(f"no labels for {stem}, skipped")
            continue
        out = out_dir / stem
        out.mkdir(parents=True, exist_ok=True)
        dets = load_or_detect(video, out)
        cuts = load_or_cut(video, out)
        gt, ignore = load_gt(lab, cuts)
        res[stem] = {}
        for name, make in TRACKERS.items():
            tracks, ms = run_tracker(make, dets.frames, cuts)
            r = evaluate(gt, tracks_to_frames(tracks, set(gt)), ignore=ignore)
            r["ms_per_frame"] = round(ms, 3)
            r["n_tracks"] = len(tracks)
            res[stem][name] = r
            print(f"{stem:45s} {name:42s} MOTA {r['mota']:.3f} IDF1 {r['idf1']:.3f} "
                  f"FP {r['fp']} FN {r['fn']} IDSW {r['idsw']} tracks {len(tracks)}")
    return res


def eval_synthetic(n: int, seed: int = 0) -> dict:
    from cinescope.track.synthetic import make_sequence

    rng = np.random.default_rng(seed)
    seqs = [make_sequence(rng, n_frames=160, cut_every=80 if i % 2 else None) for i in range(n)]
    res = {}
    for name, make in TRACKERS.items():
        tot = {"tp": 0, "fp": 0, "fn": 0, "idsw": 0, "num_gt": 0}
        idf1, ms = [], []
        for gt, dets, cuts in seqs:
            if make == "nocuts":
                tracks, t = run_tracker("nocuts", dets, cuts)
            else:
                tracks, t = run_tracker(make, dets, cuts)
            r = evaluate(dict(enumerate(gt)), tracks_to_frames(tracks))
            for k in tot:
                tot[k] += r[k]
            idf1.append(r["idf1"])
            ms.append(t)
        r = {"mota": 1 - (tot["fp"] + tot["fn"] + tot["idsw"]) / tot["num_gt"],
             "idf1_mean": float(np.mean(idf1)), **tot, "ms_per_frame": round(float(np.mean(ms)), 3)}
        res[name] = r
        print(f"synthetic x{n:<4d} {name:42s} MOTA {r['mota']:.3f} IDF1 {r['idf1_mean']:.3f} "
              f"FP {tot['fp']} FN {tot['fn']} IDSW {tot['idsw']}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--synthetic", type=int, default=0, help="number of synthetic sequences")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--save", default=str(ROOT / "p3_tracking" / "results" / "mot.json"))
    args = ap.parse_args(argv)
    save = Path(args.save)
    res = json.loads(save.read_text()) if save.exists() else {}
    if args.clips:
        res["clips"] = {**res.get("clips", {}), **eval_clips(args.clips, Path(args.out))}
    if args.synthetic:
        res["synthetic"] = {"n_sequences": args.synthetic, **eval_synthetic(args.synthetic)}
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
