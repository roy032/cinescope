"""Synthetic tracking sequences with known identities.

Real labelled tracking data for faces in films is scarce, and tests must run
without footage, so this generates the failure modes a face tracker meets:
boxes that move and change size, cross each other, miss detections for a few
frames (occlusion), jitter, drop in confidence (profile turns), and false
positives. The ground truth is exact, so tracker variants can be compared on
hundreds of sequences in seconds.
"""
from __future__ import annotations

import numpy as np


def make_sequence(rng: np.random.Generator, n_frames: int = 120, n_objects: int = 4,
                  width: int = 960, height: int = 540, miss_rate: float = 0.08,
                  occlusion_rate: float = 0.01, fp_rate: float = 0.3, jitter: float = 0.04,
                  low_score_rate: float = 0.15, speed: float = 6.0, cut_every: int | None = None):
    """Returns (gt, dets, cuts):
    gt[k]   = (ids, boxes xyxy) of frame k,
    dets[k] = (boxes xyxy, scores) the "detector" reports,
    cuts    = set of frames where a new shot starts (identities restart).
    """
    gt, dets, cuts = [], [], set()
    next_id = 0

    def spawn():
        nonlocal next_id
        h = rng.uniform(40, 160)
        o = {"id": next_id, "c": rng.uniform([h, h], [width - h, height - h]),
             "v": rng.normal(0, speed, 2), "h": h, "dh": rng.normal(0, 0.5),
             "life": int(rng.integers(n_frames // 3, n_frames * 2)), "occl": 0}
        next_id += 1
        return o

    objs = [spawn() for _ in range(n_objects)]
    for k in range(n_frames):
        if cut_every and k and k % cut_every == 0:
            cuts.add(k)
            objs = [spawn() for _ in range(int(rng.integers(1, n_objects + 1)))]
        ids, boxes, db, ds = [], [], [], []
        alive = []
        for o in objs:
            o["life"] -= 1
            o["v"] += rng.normal(0, speed * 0.15, 2)
            o["v"] = np.clip(o["v"], -2 * speed, 2 * speed)
            o["c"] = o["c"] + o["v"]
            o["h"] = float(np.clip(o["h"] + o["dh"], 25, 220))
            inside = (0 < o["c"][0] < width) and (0 < o["c"][1] < height)
            if o["life"] <= 0 or not inside:
                continue
            alive.append(o)
            w = 0.8 * o["h"]
            b = np.r_[o["c"] - [w / 2, o["h"] / 2], o["c"] + [w / 2, o["h"] / 2]]
            ids.append(o["id"])
            boxes.append(b)
            if o["occl"] > 0:
                o["occl"] -= 1
                continue
            if rng.random() < occlusion_rate:
                o["occl"] = int(rng.integers(3, 15))
                continue
            if rng.random() < miss_rate:
                continue
            noise = rng.normal(0, jitter * o["h"], 4)
            db.append(b + noise)
            ds.append(rng.uniform(0.3, 0.6) if rng.random() < low_score_rate else rng.uniform(0.6, 0.99))
        while len(alive) < n_objects and rng.random() < 0.05:
            alive.append(spawn())
        objs = alive
        for _ in range(rng.poisson(fp_rate)):
            h = rng.uniform(25, 100)
            c = rng.uniform([0, 0], [width, height])
            db.append(np.r_[c - h / 2, c + h / 2])
            ds.append(rng.uniform(0.3, 0.75))
        gt.append((ids, np.asarray(boxes).reshape(-1, 4)))
        dets.append((np.asarray(db).reshape(-1, 4), np.asarray(ds)))
    return gt, dets, cuts
