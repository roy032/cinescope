"""SubRip (.srt) cues: read, write, schedule, and score a prediction.

Scoring a predicted .srt against the true one has two parts:

  * matching — each true cue is paired with at most one predicted cue, the
    one it overlaps most in time (greedy by overlap, one-to-one). A true cue
    with no overlapping prediction is *missed*; a prediction matched to
    nothing is a *false* cue.
  * per matched pair — the text (CER, WER) and the timing (|start error|,
    |end error| in seconds).

The headline "subtitle CER" counts a missed cue as deleting all of its
characters and a false cue as inserting all of its characters, so a reader
cannot look good by skipping hard cues or by printing noise:

    CER = (edits on matched + chars of missed + chars of false) / chars of truth
"""
from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .metrics import levenshtein


@dataclass
class Cue:
    start: float
    end: float
    text: str                 # lines joined by "\n"

    @property
    def flat(self) -> str:
        return " ".join(self.text.split())


def _ts(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_ts(s: str) -> float:
    h, m, rest = s.strip().replace(".", ",").split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def write_srt(cues: Sequence[Cue], path=None) -> str:
    blocks = [f"{i}\n{_ts(c.start)} --> {_ts(c.end)}\n{c.text}\n" for i, c in enumerate(cues, 1)]
    text = "\n".join(blocks)
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def parse_srt(text: str) -> list[Cue]:
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "").strip()):
        lines = block.split("\n")
        for k, line in enumerate(lines):
            if "-->" in line:
                a, b = line.split("-->")
                body = "\n".join(lines[k + 1:]).strip()
                body = re.sub(r"<[^>]+>|\{[^}]+\}", "", body)        # <i>, {\an8} tags
                cues.append(Cue(_parse_ts(a), _parse_ts(b.split()[0]), body))
                break
    return cues


def read_srt(path) -> list[Cue]:
    with open(path, encoding="utf-8-sig") as f:
        return parse_srt(f.read())


def wrap(text: str, max_line: int = 38) -> str:
    """Break a long subtitle into two lines at the space nearest the middle."""
    if len(text) <= max_line:
        return text
    mid = len(text) // 2
    spaces = [i for i, c in enumerate(text) if c == " "]
    if not spaces:
        return text
    cut = min(spaces, key=lambda i: abs(i - mid))
    return text[:cut] + "\n" + text[cut + 1:]


def schedule(lines: Sequence[str], duration: float, rng: random.Random, start: float = 0.5,
             min_dur: float = 1.2, max_dur: float = 3.2, min_gap: float = 0.25,
             max_gap: float = 1.2, chars_per_s: float = 17.0) -> list[Cue]:
    """Lay subtitle lines on a timeline the way a subtitler would: each cue on
    screen for about its reading time (17 characters/s, clamped), with short
    gaps; lines are used in order until the clip ends."""
    cues, t = [], start
    for text in lines:
        d = min(max_dur, max(min_dur, len(text) / chars_per_s)) * rng.uniform(0.9, 1.1)
        if t + d > duration - 0.2:
            break
        cues.append(Cue(round(t, 3), round(t + d, 3), wrap(text)))
        t += d + rng.uniform(min_gap, max_gap)
    return cues


def match(truth: Sequence[Cue], pred: Sequence[Cue]) -> list[tuple[int, int]]:
    """One-to-one pairs (i_truth, j_pred) by largest temporal overlap first."""
    pairs = []
    for i, a in enumerate(truth):
        for j, b in enumerate(pred):
            ov = min(a.end, b.end) - max(a.start, b.start)
            if ov > 0:
                pairs.append((ov, i, j))
    pairs.sort(reverse=True)
    used_i, used_j, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            out.append((i, j))
    return sorted(out)


def score(truth: Sequence[Cue], pred: Sequence[Cue], norm=lambda s: s) -> dict:
    """CER/WER and timing of a predicted cue list. `norm` maps both texts
    before comparison (e.g. charset.fold for case- and punctuation-blind CER)."""
    pairs = match(truth, pred)
    mi = {i for i, _ in pairs}
    mj = {j for _, j in pairs}
    t_text = [norm(c.flat) for c in truth]
    p_text = [norm(c.flat) for c in pred]
    edits = sum(levenshtein(t_text[i], p_text[j]) for i, j in pairs)
    missed = sum(len(t_text[i]) for i in range(len(truth)) if i not in mi)
    false = sum(len(p_text[j]) for j in range(len(pred)) if j not in mj)
    n_chars = max(1, sum(len(t) for t in t_text))
    w_edits = sum(levenshtein(t_text[i].split(), p_text[j].split()) for i, j in pairs)
    w_missed = sum(len(t_text[i].split()) for i in range(len(truth)) if i not in mi)
    w_false = sum(len(p_text[j].split()) for j in range(len(pred)) if j not in mj)
    n_words = max(1, sum(len(t.split()) for t in t_text))
    ds = [abs(truth[i].start - pred[j].start) for i, j in pairs]
    de = [abs(truth[i].end - pred[j].end) for i, j in pairs]
    exact = sum(t_text[i] == p_text[j] for i, j in pairs)
    return {
        "cues": len(truth), "predicted": len(pred), "matched": len(pairs),
        "recall": len(pairs) / max(1, len(truth)), "precision": len(pairs) / max(1, len(pred)),
        "cer": (edits + missed + false) / n_chars,
        "cer_matched": edits / max(1, sum(len(t_text[i]) for i, _ in pairs)),
        "wer": (w_edits + w_missed + w_false) / n_words,
        "exact_cues": exact / max(1, len(truth)),
        "start_err_s": sum(ds) / max(1, len(ds)), "end_err_s": sum(de) / max(1, len(de)),
        "chars": n_chars, "edits": edits, "missed_chars": missed, "false_chars": false,
    }
