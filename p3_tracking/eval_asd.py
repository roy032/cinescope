"""Score who's-talking detection against hand-labelled speaking intervals (Week 11).

    pip install silero-vad          # optional: the library VAD row
    python p3_tracking/eval_asd.py trailers/goldeneye.mp4

Every face-frame of a track whose identity is labelled (labels/<clip>.ids.json)
is one example; it is positive if the time falls in a speaking interval of
that identity (labels/<clip>.speech.json), and skipped inside an "unsure"
interval. Systems:

  ours                 audio VAD x mouth motion x synchrony (cinescope/speaker/active.py)
  ours, parts          each cue alone, to see what carries the score
  ours, Silero gate    our visual cues, with Silero's speech probability as the audio cue
  Light-ASD            the AVA-trained network (cinescope/speaker/lightasd.py)

Thresholds for F1 are fixed in advance: 0.25 for ours (e.g. speech 1, mouth
0.5, no synchrony bonus), 0 for Light-ASD's logit (its authors' demo).
-> p3_tracking/results/asd.json
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

from cinescope.speaker.active import score_track  # noqa: E402
from cinescope.speaker.audio import load_audio, silero_vad, vad  # noqa: E402
from cinescope.speaker.metrics import average_precision, f1_at, roc_auc  # noqa: E402
from p3_tracking.eval_cluster import gt_identity  # noqa: E402

THRESH = {"ours": 0.25, "Light-ASD": 0.0}


def in_any(t: np.ndarray, ivs: list[dict], ident: str) -> np.ndarray:
    m = np.zeros(len(t), bool)
    for iv in ivs:
        if iv["id"] == ident:
            m |= (t >= iv["t0"]) & (t < iv["t1"])
    return m


def main(argv=None):
    from p3_tracking.track_clip import load_or_detect
    from p3_tracking.train_embed import clip_crops

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--save", default=str(ROOT / "p3_tracking" / "results" / "asd.json"))
    ap.add_argument("--no-lightasd", action="store_true")
    args = ap.parse_args(argv)
    res = {}
    for video in args.videos:
        stem = Path(video).stem
        ids = json.loads((ROOT / "p3_tracking" / "labels" / f"{stem}.ids.json").read_text())
        sp = json.loads((ROOT / "p3_tracking" / "labels" / f"{stem}.speech.json").read_text())
        labels = {f["index"]: f["boxes"] for f in ids["frames"]}
        crops = clip_crops(video, Path(args.out), 24)
        dets = load_or_detect(video, Path(args.out) / stem)
        times, fps = dets.times, dets.fps
        audio = load_audio(video)
        t0 = time.perf_counter()
        v = vad(audio)
        t_vad = time.perf_counter() - t0
        sil = silero_vad(audio)
        lasd = None
        if not args.no_lightasd:
            from cinescope.speaker.lightasd import LightASD

            lasd = LightASD()
        cols: dict[str, list] = {k: [] for k in ("y", "ours", "speech", "mouth", "sync", "ours_nosync",
                                                  "silero_gate", "Light-ASD", "track")}
        secs = {"ours": t_vad, "Light-ASD": 0.0}
        for c in crops:
            ident = gt_identity(c.track, labels)
            if ident is None:
                continue
            t = times[np.asarray(c.track.frames)]
            keep = ~in_any(t, sp.get("unsure", []), ident)
            y = in_any(t, sp["intervals"], ident)
            a = time.perf_counter()
            r = score_track(c.gray, c.track.frames, times, v, fps)
            secs["ours"] += time.perf_counter() - a
            p_mouth = 1 / (1 + np.exp(-(r["mouth_z"] - 1.0)))
            cols["y"] += y[keep].tolist()
            cols["ours"] += r["score"][keep].tolist()
            cols["speech"] += r["speech"][keep].tolist()
            cols["mouth"] += p_mouth[keep].tolist()
            cols["sync"] += r["sync"][keep].tolist()
            cols["ours_nosync"] += (r["speech"] * p_mouth)[keep].tolist()
            if sil is not None:
                vs = {"fps": 100.0, "score": sil, "energy_db": v["energy_db"][:len(sil)]}
                rs = score_track(c.gray, c.track.frames, times, vs, fps)
                cols["silero_gate"] += rs["score"][keep].tolist()
            if lasd is not None:
                a = time.perf_counter()
                cols["Light-ASD"] += lasd.score(c.asd, t, audio)[keep].tolist()
                secs["Light-ASD"] += time.perf_counter() - a
            cols["track"] += [c.track.id] * int(keep.sum())
        y = np.asarray(cols["y"], bool)
        out = {"face_frames": int(len(y)), "speaking": int(y.sum()), "seconds": {k: round(s, 1) for k, s in secs.items()},
               "systems": {}}
        names = {"ours": "ours", "ours_nosync": "ours without synchrony", "speech": "ours, audio VAD only",
                 "mouth": "ours, mouth motion only", "sync": "ours, synchrony only",
                 "silero_gate": "ours with Silero VAD as audio cue", "Light-ASD": "Light-ASD (AVA)"}
        for k, name in names.items():
            s = np.asarray(cols[k], float)
            if len(s) != len(y):
                continue
            r = {"AP": average_precision(y, s), "AUC": roc_auc(y, s)}
            if k in THRESH:
                r.update({f"{m}@{THRESH[k]}": v for m, v in f1_at(y, s, THRESH[k]).items()})
            out["systems"][name] = r
            print(f"{stem:12s} {name:36s} AP {r['AP']:.3f} AUC {r['AUC']:.3f}"
                  + (f" F1@{THRESH[k]} {r[f'f1@{THRESH[k]}']:.3f}" if k in THRESH else ""))
        # Audio-only: our VAD against Silero, frame by frame (agreement, not accuracy).
        if sil is not None:
            n = min(len(sil), len(v["score"]))
            a, b = v["score"][:n] >= 0.5, sil[:n] >= 0.5
            out["vad_agreement_with_silero"] = {"ours_speech_fraction": float(a.mean()),
                                                 "silero_speech_fraction": float(b.mean()),
                                                 "agreement": float((a == b).mean()),
                                                 "ours_recall_of_silero": float((a & b).sum() / max(b.sum(), 1))}
            print(out["vad_agreement_with_silero"])
        print(f"{stem}: {len(y)} face-frames scored, {int(y.sum())} speaking")
        res[stem] = out
    save = Path(args.save)
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
