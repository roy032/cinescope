"""Score the event detector on synthetic scenes with known events.

    python p6_pose_video/eval_events.py --tune            # grid-search thresholds on the tuning seeds
    python p6_pose_video/eval_events.py                   # report on the separate test seeds
    python p6_pose_video/eval_events.py --clip outputs/goldeneye/breakdown.json outputs/goldeneye_yolo/goldeneye/breakdown.json \
        --labels p6_pose_video/labels/goldeneye_events.json --out p6_pose_video/results/events_goldeneye.json

Scenes come from cinescope/pose/synthetic.py: fights (punches, kicks, some
ending in a fall), conversations with hand gestures (people close together,
the trap for "strike"), chases, walkers who fall, plain walking, idling; a
third of them with a camera pan. Tracking runs on the synthetic detections
exactly as on real ones. Thresholds are chosen on seeds 0-199 (`--tune`)
and reported on seeds 1000-1199.

`--clip` scores the events of breakdown.py runs on a real clip against by-eye
labels (p6_pose_video/labels/<clip>_events.json): precision / recall where the
clip has events, and false alarms per minute, with the ones that fall on the
labelled hard negatives (fast movements that are not strikes) listed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.pose.events import EventParams, detect_events, match_events  # noqa: E402
from cinescope.pose.synthetic import random_scene  # noqa: E402
from cinescope.pose.track import track_people  # noqa: E402


def make_scenes(seeds):
    return [random_scene(np.random.default_rng(s)) for s in seeds]


def score(scenes, p: EventParams) -> dict:
    tot: dict[str, dict] = {}
    for sc in scenes:
        tracks = track_people(sc.times, sc.boxes, sc.kpts)
        pred = [e.to_dict() for e in detect_events(tracks, p)]
        r = match_events(pred, sc.events)
        for typ, m in r.items():
            t = tot.setdefault(typ, {"tp": 0, "fp": 0, "fn": 0})
            for k in ("tp", "fp", "fn"):
                t[k] += m[k]
    for t in tot.values():
        p_ = t["tp"] / max(1, t["tp"] + t["fp"])
        r_ = t["tp"] / max(1, t["tp"] + t["fn"])
        t.update(precision=round(p_, 3), recall=round(r_, 3), f1=round(2 * p_ * r_ / max(1e-9, p_ + r_), 3))
    return tot


def macro_f1(res: dict) -> float:
    return float(np.mean([res[k]["f1"] for k in ("strike", "fall", "run") if k in res]))


def score_clip(breakdowns: list[str], labels: str) -> dict:
    lab = json.loads(Path(labels).read_text(encoding="utf-8"))
    out = {"labels": labels, "n_labelled_events": len(lab["events"])}
    for b in breakdowns:
        d = json.loads(Path(b).read_text(encoding="utf-8"))
        pred = [e for s in d["shots"] for e in s["events"]]
        minutes = (d["shots"][-1]["t1"] - d["shots"][0]["t0"]) / 60 if d["shots"] else 0.0
        on_neg = [e for e in pred if any(h["t0"] - 0.5 <= e["t"] <= h["t1"] + 0.5 for h in lab["hard_negatives"])]
        out[f"pose={d['pose_model']}"] = {
            "breakdown": b, "minutes": round(minutes, 2), "predicted": len(pred),
            "false_alarms_per_minute": round((len(pred) - sum(m["tp"] for m in match_events(pred, lab["events"])
                                                               .values())) / max(minutes, 1e-9), 2),
            "match": {t: {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in m.items()}
                      for t, m in match_events(pred, lab["events"]).items()},
            "on_hard_negatives": len(on_neg),
            "events": [{k: e[k] for k in ("type", "t", "t0", "t1", "score")} for e in pred]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--clip", nargs="+", help="breakdown.json files of one clip (score against --labels)")
    ap.add_argument("--labels")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.clip:
        res = score_clip(args.clip, args.labels)
        print(json.dumps(res, indent=1, default=str))
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        return
    args.out = args.out or "p6_pose_video/results/events_synthetic.json"
    p = EventParams()
    if args.tune:
        scenes = make_scenes(range(args.n))
        best = (-1, p)
        grid = itertools.product([0.8, 1.0, 1.2, 1.6], [0.05, 0.1, 0.15, 0.25], [1.2, 1.5, 1.8], [0.05, 0.1],
                                 [0.4, 0.6, 1.0, 99.0])
        for vs, dc, vr, hd, ps in grid:
            q = replace(p, v_strike=vs, d_contact=dc, v_run=vr, hip_drop=hd, max_pair_speed=ps)
            f = macro_f1(score(scenes, q))
            if f > best[0]:
                best = (f, q)
        print("best tuning macro-F1", round(best[0], 3), asdict(best[1]))
        return
    tune = score(make_scenes(range(args.n)), p)
    test = score(make_scenes(range(1000, 1000 + args.n)), p)
    res = {"params": asdict(p), "tuning_seeds": tune, "test_seeds": test,
           "test_macro_f1": round(macro_f1(test), 3), "n_scenes_each": args.n}
    print(json.dumps(res, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
