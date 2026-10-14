"""Action events from pose tracks: strikes, falls, running.

Everything is measured in body heights (the track's median box size), so a
punch thrown by a figure 100 px tall and one 500 px tall look the same, and
in seconds, so the sampling rate does not matter.

Camera motion. A film camera pans, tracks and shakes, so a keypoint's speed
in the image mixes the person's motion with the camera's. Limb speeds are
therefore measured *relative to the person's own hip centre*: a pan moves hip
and wrist alike and cancels. That is why "running" is detected from the legs
(ankles swinging fast relative to the hips) rather than from the body moving
across the frame, which a tracking shot hides completely and a pan fakes.

  strike   a wrist or ankle moves faster than `v_strike` body heights/s
           relative to its hips, towards another person, ending within
           `d_contact` body heights of that person's box, while the two
           people's hips are not moving past each other (relative speed
           below `max_pair_speed`; also camera-independent). Contact itself is
           not observable in 2-D (depth is unknown), so "strike" means "fast
           limb that reaches someone".
  fall     the torso goes from near-upright (< `upright_deg`) to near-
           horizontal (> `down_deg`) within `fall_window` seconds while the
           hips drop by >= `hip_drop` body heights.
  run      the legs' relative speed stays above `v_run` for at least
           `min_run` seconds (median over a sliding window).

Thresholds are chosen on synthetic scenes (cinescope/pose/synthetic.py,
p6_pose_video/eval_events.py) and never on the hand-labelled clip.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np

from . import KP
from .track import Track


@dataclass
class EventParams:
    min_conf: float = 0.3          # keypoint confidence to use it
    max_gap: float = 0.5           # s; longer gaps break speed estimates
    v_strike: float = 1.0          # body heights / s, limb relative to hips
    d_contact: float = 0.10        # body heights from the other person's box
    strike_merge: float = 0.4      # s; detections closer than this are one strike
    max_pair_speed: float = 0.6    # body heights / s; two people moving past each other are not fighting
    v_run: float = 1.5             # body heights / s, mean ankle speed relative to hips
    run_window: float = 0.75       # s, sliding median window
    min_run: float = 0.75          # s
    upright_deg: float = 35.0
    down_deg: float = 60.0
    fall_window: float = 1.2       # s
    hip_drop: float = 0.05         # body heights


@dataclass
class Event:
    type: str
    t: float                       # peak time
    t0: float
    t1: float
    track: int
    other: int | None = None
    score: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("t", "t0", "t1", "score"):
            d[k] = round(float(d[k]), 3)
        return d


def _pt(k: np.ndarray, names: tuple[str, ...], min_conf: float) -> np.ndarray:
    """Mean of the named keypoints that are confident, per frame: (T, 2), NaN if none."""
    idx = [KP[n] for n in names]
    xy, c = k[:, idx, :2], k[:, idx, 2] >= min_conf
    w = c.astype(np.float64)
    s = w.sum(1)
    out = (xy * w[..., None]).sum(1) / np.maximum(s, 1)[:, None]
    out[s == 0] = np.nan
    return out


@dataclass
class TrackFeatures:
    id: int
    t: np.ndarray                  # (T,)
    boxes: np.ndarray              # (T, 4)
    scale: float                   # body height, px
    hip: np.ndarray                # (T, 2)
    torso_deg: np.ndarray          # (T,) angle from vertical, NaN if unknown
    limbs: np.ndarray              # (T, 4, 2) wrists L/R, ankles L/R (NaN if unknown)
    limb_speed: np.ndarray         # (T, 4) relative to hips, body heights / s (NaN at t0 / gaps)


def features(tr: Track, p: EventParams) -> TrackFeatures:
    t, b, k = tr.arrays()
    size = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])       # max(w, h): also right when lying down
    scale = float(np.median(size))
    hip = _pt(k, ("left_hip", "right_hip"), p.min_conf)
    box_c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    hip = np.where(np.isnan(hip), box_c, hip)
    sh = _pt(k, ("left_shoulder", "right_shoulder"), p.min_conf)
    v = sh - hip
    torso = np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1]))     # 0 = upright, 90 = lying, 180 = upside down
    limbs = np.stack([_pt(k, (n,), p.min_conf) for n in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")], 1)
    rel = limbs - hip[:, None, :]
    speed = np.full((len(t), 4), np.nan)
    if len(t) > 1:
        dt = np.diff(t)
        d = np.linalg.norm(np.diff(rel, axis=0), axis=-1) / dt[:, None] / scale
        d[dt > p.max_gap] = np.nan
        speed[1:] = d
    return TrackFeatures(tr.id, t, b, scale, hip, torso, limbs, speed)


def _index_at(f: TrackFeatures, t: float, tol: float) -> int | None:
    i = int(np.argmin(np.abs(f.t - t)))
    return i if abs(f.t[i] - t) <= tol else None


def _dist_to_box(pt: np.ndarray, box: np.ndarray) -> float:
    dx = max(box[0] - pt[0], 0, pt[0] - box[2])
    dy = max(box[1] - pt[1], 0, pt[1] - box[3])
    return float(np.hypot(dx, dy))


def detect_strikes(feats: list[TrackFeatures], p: EventParams) -> list[Event]:
    raw: list[Event] = []
    for f in feats:
        for i in range(1, len(f.t)):
            for e in range(4):
                s = f.limb_speed[i, e]
                if not np.isfinite(s) or s < p.v_strike:
                    continue
                end = f.limbs[i, e]
                for g in feats:
                    if g.id == f.id:
                        continue
                    j = _index_at(g, f.t[i], p.max_gap / 2)
                    jp = _index_at(g, f.t[i - 1], p.max_gap / 2)
                    if j is None:
                        continue
                    box = g.boxes[j]
                    if _dist_to_box(end, box) > p.d_contact * f.scale:
                        continue
                    # the pair's relative motion (camera-independent): people walking past
                    # each other swing their limbs past each other's boxes too
                    if jp is not None and jp != j:
                        rel = (f.hip[i] - g.hip[j]) - (f.hip[i - 1] - g.hip[jp])
                        if np.linalg.norm(rel) / (f.t[i] - f.t[i - 1]) / f.scale > p.max_pair_speed:
                            continue
                    # moving towards the other person (relative to own hips)
                    mv = (f.limbs[i, e] - f.hip[i]) - (f.limbs[i - 1, e] - f.hip[i - 1])
                    to = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]) - f.limbs[i - 1, e]
                    if not np.all(np.isfinite(mv)) or float(mv @ to) <= 0:
                        continue
                    raw.append(Event("strike", float(f.t[i]), float(f.t[i - 1]), float(f.t[i]), f.id, g.id, float(s)))
    raw.sort(key=lambda ev: ev.t)
    merged: list[Event] = []
    for ev in raw:
        if merged and ev.t - merged[-1].t1 <= p.strike_merge:
            m = merged[-1]
            m.t1 = max(m.t1, ev.t1)
            if ev.score > m.score:
                m.t, m.score, m.track, m.other = ev.t, ev.score, ev.track, ev.other
            continue
        merged.append(ev)
    return merged


def detect_falls(feats: list[TrackFeatures], p: EventParams) -> list[Event]:
    out = []
    for f in feats:
        armed_until = -np.inf
        for i in range(len(f.t)):
            if not np.isfinite(f.torso_deg[i]) or f.torso_deg[i] < p.down_deg or f.t[i] < armed_until:
                continue
            win = (f.t >= f.t[i] - p.fall_window) & (f.t < f.t[i])
            up = win & (f.torso_deg < p.upright_deg)
            if not up.any():
                continue
            j = int(np.where(up)[0][-1])
            drop = (f.hip[i, 1] - f.hip[j, 1]) / f.scale
            if drop < p.hip_drop:
                continue
            out.append(Event("fall", float(f.t[i]), float(f.t[j]), float(f.t[i]), f.id, None,
                             float(f.torso_deg[i] - f.torso_deg[j])))
            armed_until = f.t[i] + 2 * p.fall_window      # one fall at a time
    return out


def detect_runs(feats: list[TrackFeatures], p: EventParams) -> list[Event]:
    out = []
    for f in feats:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows (ankles unseen) stay NaN
            leg = np.nanmean(f.limb_speed[:, 2:], axis=1) if len(f.t) else np.zeros(0)
        n = len(f.t)
        sm = np.full(n, np.nan)
        for i in range(n):
            w = (f.t >= f.t[i] - p.run_window / 2) & (f.t <= f.t[i] + p.run_window / 2)
            vals = leg[w]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2:
                sm[i] = np.median(vals)
        on = np.nan_to_num(sm, nan=0.0) >= p.v_run
        i = 0
        while i < n:
            if not on[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and on[j + 1] and f.t[j + 1] - f.t[j] <= p.max_gap:
                j += 1
            if f.t[j] - f.t[i] >= p.min_run:
                pk = i + int(np.nanargmax(sm[i:j + 1]))
                out.append(Event("run", float(f.t[pk]), float(f.t[i]), float(f.t[j]), f.id, None, float(sm[pk])))
            i = j + 1
    return out


def detect_events(tracks: list[Track], p: EventParams | None = None) -> list[Event]:
    p = p or EventParams()
    feats = [features(tr, p) for tr in tracks if len(tr) >= 2]
    ev = detect_strikes(feats, p) + detect_falls(feats, p) + detect_runs(feats, p)
    return sorted(ev, key=lambda e: e.t)


def match_events(pred: list[dict], gt: list[dict], tol: float = 0.5) -> dict:
    """Per event type: TP / FP / FN, precision, recall, F1. A prediction matches a
    ground-truth event of the same type when its peak time lies within the
    event's span widened by `tol` s, or (runs) when the spans overlap. One-to-one."""
    out = {}
    for typ in sorted({e["type"] for e in pred} | {e["type"] for e in gt}):
        P = sorted([e for e in pred if e["type"] == typ], key=lambda e: -e.get("score", 0))
        G = [e for e in gt if e["type"] == typ]
        used = [False] * len(G)
        tp = 0
        for e in P:
            best, bj = None, -1
            for j, g in enumerate(G):
                if used[j]:
                    continue
                if typ == "run":
                    ok = e["t0"] <= g["t1"] + tol and e["t1"] >= g["t0"] - tol
                else:
                    ok = g["t0"] - tol <= e["t"] <= g["t1"] + tol
                if ok:
                    d = abs(e["t"] - 0.5 * (g["t0"] + g["t1"]))
                    if best is None or d < best:
                        best, bj = d, j
            if bj >= 0:
                used[bj] = True
                tp += 1
        fp, fn = len(P) - tp, len(G) - tp
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
        out[typ] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
    return out
