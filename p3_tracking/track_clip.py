"""Detect and track faces through a clip (Week 9).

    python p3_tracking/track_clip.py trailers/goldeneye.mp4
    python p3_tracking/track_clip.py trailers/goldeneye.mp4 --tracker bytetrack     # ultralytics
    python p3_tracking/track_clip.py clip.mp4 --detector centernet --ckpt p2_detection/checkpoints/centernet_r18.pt

Writes outputs/<clip>/faces_<detector>.npz (per-frame detections, reused by
every later step), shots.json (Phase 0 cuts) and tracks_<tracker>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.faces.clip import detect_clip, shot_starts, track_clip  # noqa: E402
from cinescope.faces.detectors import make_detector  # noqa: E402


def load_or_detect(video: str, out: Path, detector: str = "yunet", ckpt: str | None = None):
    cache = out / f"faces_{detector}.npz"
    det = None if cache.exists() else make_detector(detector, ckpt=ckpt)
    return detect_clip(video, det, cache=cache)


def load_or_cut(video: str, out: Path) -> list[int]:
    p = out / "shots.json"
    if p.exists():
        return json.loads(p.read_text())["starts"]
    starts = shot_starts(video)
    p.write_text(json.dumps({"video": video, "starts": starts}))
    return starts


def make_tracker(kind: str):
    if kind == "ours":
        from cinescope.track.tracker import Tracker

        return Tracker()
    from cinescope.track.ultra import UltralyticsTracker

    return UltralyticsTracker(kind, **({"gmc_method": None} if kind == "botsort" else {}))


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--detector", default="yunet", choices=["yunet", "centernet"])
    ap.add_argument("--ckpt", help="Phase 2 CenterNet checkpoint (for --detector centernet)")
    ap.add_argument("--tracker", default="ours", choices=["ours", "bytetrack", "botsort"])
    ap.add_argument("--no-cuts", action="store_true", help="do not reset tracks at shot cuts")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    out = Path(args.out) / Path(args.video).stem
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    dets = load_or_detect(args.video, out, args.detector, args.ckpt)
    t1 = time.perf_counter()
    cuts = [] if args.no_cuts else load_or_cut(args.video, out)
    t2 = time.perf_counter()
    tracks = track_clip(dets, cuts, tracker=make_tracker(args.tracker))
    t3 = time.perf_counter()
    res = {"video": args.video, "detector": args.detector, "tracker": args.tracker, "fps": dets.fps,
           "n_frames": len(dets.frames), "n_shots": len(cuts), "n_tracks": len(tracks),
           "seconds": {"detect": round(t1 - t0, 1), "cuts": round(t2 - t1, 1), "track": round(t3 - t2, 2)},
           "tracks": [t.to_dict() for t in tracks]}
    suffix = args.tracker + ("_nocuts" if args.no_cuts else "")
    (out / f"tracks_{suffix}.json").write_text(json.dumps(res))
    print(f"{args.video}: {len(dets.frames)} frames, {sum(len(d) for d in dets.frames)} detections, "
          f"{len(cuts)} shots, {len(tracks)} tracks ({args.tracker}); "
          f"tracking {1000 * (t3 - t2) / max(len(dets.frames), 1):.2f} ms/frame")
    return res


if __name__ == "__main__":
    main()
