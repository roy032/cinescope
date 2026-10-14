"""Credits and other on-screen text: rolling or static, anywhere in the frame.

Subtitles have a known place (bottom, centred) and stay still. Credits do
not: a title card sits anywhere, and end credits roll upward at a constant
speed. The reader therefore works in *content* coordinates:

  1. text lines over the whole frame, either polarity (cinescope.ocr.textmap);
  2. the vertical scroll between consecutive samples, from the
     cross-correlation of the two frames' stroke row profiles (a 1-D
     search: credits only move vertically). The running sum turns a line's
     screen y into its position on the virtual credits "page";
  3. readings of the same line in successive samples land at the same page
     position, so they are grouped into tracks (page y within half a line,
     overlapping x, no long time gap); static title cards are the zero-scroll
     special case;
  4. each track with enough readings is one credit line; its text is the
     reading the recogniser was most confident about.

Persistence (min_reads) and confidence filter the texture that the text map
mistakes for text in ordinary scenes.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .metrics import cer
from .subtitles import _resize, reader_threshold
from .textmap import TextParams, detect_lines, pad_box

ReadFn = Callable[[Sequence[np.ndarray]], list[tuple[str, float]]]


@dataclass
class CreditsParams:
    work_h: int = 360
    max_scroll: float = 0.25        # largest scroll per sample, fraction of the height
    min_reads: int = 2              # samples a line must be seen in
    max_reads: int = 3              # readings per line (each costs a recogniser call)
    min_conf: float | None = None   # None: the reader's own credits_min_conf
    max_gap_samples: int = 2
    dedupe_s: float = 10.0          # the same text again within this many seconds is a repeat
    text: TextParams = field(default_factory=lambda: TextParams(polarity="both"))


@dataclass
class Track:
    page_y: float
    x0: float
    x1: float
    h: float
    first_t: float
    last_t: float
    reads: list = field(default_factory=list)       # (text, conf)
    seen: int = 0

    def best(self) -> tuple[str, float]:
        return max(self.reads, key=lambda r: r[1])


def scroll_shift(prev: np.ndarray, cur: np.ndarray, max_shift: int) -> int:
    """Vertical shift s (rows) maximising the normalised correlation of
    prev[y + s] with cur[y] over their overlap: positive when content moved up.

    All shifts at once: the raw products come from one np.correlate, the
    overlap energies from cumulative sums. Ties go to the smallest |s|, so a
    blank frame reports no motion."""
    a = np.asarray(prev, np.float64) - np.mean(prev)
    b = np.asarray(cur, np.float64) - np.mean(cur)
    n = len(a)
    m = min(max_shift, n - 1)
    shifts = np.arange(-m, m + 1)
    raw = np.correlate(a, b, mode="full")[shifts + n - 1]          # sum_y a[y+s] b[y]
    ca = np.concatenate([[0.0], np.cumsum(a * a)])                 # prefix energies
    cb = np.concatenate([[0.0], np.cumsum(b * b)])
    ea = np.where(shifts >= 0, ca[n] - ca[np.clip(shifts, 0, n)], ca[np.clip(n + shifts, 0, n)])
    eb = np.where(shifts >= 0, cb[np.clip(n - shifts, 0, n)], cb[n] - cb[np.clip(-shifts, 0, n)])
    d = np.sqrt(ea * eb)
    c = np.where(d > 0, raw / np.where(d > 0, d, 1), -np.inf)
    best = c.max()
    if not np.isfinite(best):
        return 0
    ties = shifts[c >= best - 1e-9]
    return int(ties[np.argmin(np.abs(ties))])


class CreditsReader:
    def __init__(self, read_fn: ReadFn, params: CreditsParams | None = None):
        self.read_fn = read_fn
        self.p = params or CreditsParams()

    def run(self, frames: Iterable[tuple[float, np.ndarray]]) -> list[dict]:
        p = self.p
        active: list[Track] = []
        done: list[Track] = []
        offset, prev_prof, period, last_t = 0.0, None, None, None
        for t, rgb in frames:
            if last_t is not None and period is None:
                period = t - last_t
            last_t = t
            h = rgb.shape[0]
            scale = min(1.0, p.work_h / h)
            small = _resize(rgb, scale)
            boxes, strokes = detect_lines(small, p.text, frame_h=small.shape[0])
            prof = strokes.sum(1).astype(np.float64)
            if prev_prof is not None:
                offset += scroll_shift(prev_prof, prof, int(p.max_scroll * small.shape[0]))
            prev_prof = prof
            gap = (p.max_gap_samples + 0.5) * (period or 1.0)
            still = []
            for tr in active:
                (still if t - tr.last_t <= gap else done).append(tr)
            active = still
            # match lines to tracks by page position first; read only the
            # crops of tracks that do not have enough readings yet
            to_read = []
            for b in boxes:
                py, x0, x1, bh = (b[1] + b[3]) / 2 + offset, b[0], b[2], b[3] - b[1]
                match = None
                for tr in active:
                    ov = min(x1, tr.x1) - max(x0, tr.x0)
                    if abs(py - tr.page_y) < 0.5 * max(bh, tr.h) and tr.last_t < t \
                            and ov > 0.5 * min(x1 - x0, tr.x1 - tr.x0):
                        match = tr
                        break
                if match is None:
                    match = Track(py, x0, x1, bh, t, t)
                    active.append(match)
                match.page_y, match.x0, match.x1, match.h = py, x0, x1, bh
                match.last_t = t
                match.seen += 1
                if len(match.reads) < p.max_reads:
                    fb = tuple(int(round(v / scale)) for v in b)
                    x0f, y0f, x1f, y1f = pad_box(fb, rgb.shape[0], rgb.shape[1])
                    to_read.append((match, rgb[y0f:y1f, x0f:x1f]))
            if to_read:
                for (tr, _), (text, conf) in zip(to_read, self.read_fn([c for _, c in to_read]), strict=True):
                    if text.strip():
                        tr.reads.append((text.strip(), conf))
        done += active
        min_conf = reader_threshold(self.read_fn, "credits_min_conf", 0.6) if p.min_conf is None else p.min_conf
        lines = []
        for tr in sorted(done, key=lambda tr: (tr.first_t, tr.page_y)):
            if not tr.reads:
                continue
            text, conf = tr.best()
            if tr.seen < p.min_reads or conf < min_conf or sum(c.isalnum() for c in text) < 2:
                continue
            if any(tr.first_t - ln["t"] <= p.dedupe_s and cer(ln["text"], text) < 0.15 for ln in lines):
                continue                     # the same text again (a static card re-detected)
            lines.append({"t": round(tr.first_t, 2), "end": round(tr.last_t, 2), "text": text,
                          "conf": round(conf, 3), "seen": tr.seen})
        return lines


def score_lines(truth: Sequence[str], pred: Sequence[str], found_cer: float = 0.2) -> dict:
    """Each true line against its closest predicted line (one-to-one, greedy by
    CER). Recall = true lines read with CER <= found_cer."""
    pairs = sorted(((cer(t, q), i, j) for i, t in enumerate(truth) for j, q in enumerate(pred)))
    ui, uj, best = set(), set(), {}
    for c, i, j in pairs:
        if i not in ui and j not in uj:
            ui.add(i)
            uj.add(j)
            best[i] = min(1.0, c)
    cers = [best.get(i, 1.0) for i in range(len(truth))]
    found = sum(c <= found_cer for c in cers)
    return {"lines": len(truth), "predicted": len(pred), "found": found,
            "recall": found / max(1, len(truth)), "precision": found / max(1, len(pred)),
            "mean_cer": float(np.mean(cers)) if cers else 0.0}
