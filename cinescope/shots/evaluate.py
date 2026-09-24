"""Scoring shot-boundary detection against hand labels.

Label file (written by the labelling tool, p0_classical/label_cuts.py):

    {"video": "Ted_Lasso_120s.mkv", "fps": 23.976, "n_frames": 7218,
     "transitions": [{"kind": "cut", "frame": 131},
                     {"kind": "dissolve", "start": 880, "end": 902}, ...]}

`frame` is the first frame of the new shot. Gradual transitions carry the span.

Matching (one-to-one, greedy by distance), as in the TRECVID SBD task:
  * a predicted boundary matches a labelled CUT if |pred - label| <= tol frames
  * it matches a labelled GRADUAL transition if it falls inside [start - tol, end + tol]
Precision = matched / predicted, recall = matched / labelled, F1 = harmonic mean.
Reported overall and separately for cuts and gradual transitions, because a
histogram detector is expected to be much better at the former.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Score:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "precision": round(self.precision, 4),
                "recall": round(self.recall, 4), "f1": round(self.f1, 4)}


def load_labels(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for tr in data["transitions"]:
        if tr["kind"] == "cut":
            tr.setdefault("start", tr["frame"])
            tr.setdefault("end", tr["frame"])
        else:
            tr.setdefault("frame", (tr["start"] + tr["end"]) // 2)
    return data


def _dist(pred: int, lab: dict, tol: int) -> float | None:
    if lab["kind"] == "cut":
        d = abs(pred - lab["frame"])
        return d if d <= tol else None
    if lab["start"] - tol <= pred <= lab["end"] + tol:
        return abs(pred - lab["frame"]) / 1000.0     # inside the span: always a good match
    return None


def match(pred_frames: list[int], labels: list[dict], tol: int = 2) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy one-to-one matching. Returns (pairs (pred_i, label_j), unmatched preds, unmatched labels)."""
    cand = []
    for i, p in enumerate(pred_frames):
        for j, lab in enumerate(labels):
            d = _dist(p, lab, tol)
            if d is not None:
                cand.append((d, i, j))
    cand.sort()
    used_p, used_l, pairs = set(), set(), []
    for _, i, j in cand:
        if i in used_p or j in used_l:
            continue
        used_p.add(i)
        used_l.add(j)
        pairs.append((i, j))
    return (pairs, [i for i in range(len(pred_frames)) if i not in used_p],
            [j for j in range(len(labels)) if j not in used_l])


def score(pred_frames: list[int], labels: list[dict], tol: int = 2) -> dict:
    pairs, fp, fn = match(pred_frames, labels, tol)
    overall = Score(len(pairs), len(fp), len(fn))
    out = {"overall": overall.as_dict()}
    for kind in ("cut", "gradual"):
        is_kind = [(lab["kind"] == "cut") == (kind == "cut") for lab in labels]
        tp = sum(1 for _, j in pairs if is_kind[j])
        miss = sum(1 for j in fn if is_kind[j])
        # a false positive has no label, so attribute it to neither kind; report
        # recall per kind and precision only overall
        out[kind] = {"n": sum(is_kind), "tp": tp, "fn": miss,
                     "recall": round(tp / (tp + miss), 4) if tp + miss else None}
    out["false_positives"] = [pred_frames[i] for i in fp]
    out["missed"] = [labels[j] for j in fn]
    return out
