"""Burned-in subtitles -> timed cues (.srt).

    frames (sampled at ~8 fps)
      -> text lines in the subtitle band (cinescope.ocr.textmap)
      -> line tracks: the same line, at the same place, with the same strokes,
         in consecutive samples
      -> one clean image per track: the per-pixel temporal median of its crops
      -> recogniser -> text; unreadable tracks (low confidence) are dropped
      -> cues: tracks that appear and disappear together are the lines of one
         subtitle
      -> de-duplication of fragments -> .srt

Two observations drive the design:

  * **A subtitle stays still and does not change while it is on screen.**
    A line in the next sample continues a track if its box overlaps the
    track's box and its stroke pixels agree with the track's (IoU of the
    stroke maps). When the subtitle changes, the strokes change. Tracking per
    line, not per frame, matters on static shots: texture that the text map
    mistakes for a line (a shelf edge, a keyboard) is just as still as a
    subtitle, and would otherwise glue consecutive subtitles into one event.
  * **The median of a static overlay over a moving background is the
    overlay.** Each pixel's median over the track's crops keeps the text
    (identical in every frame) and flattens the background, so the
    recogniser reads a cleaner image than any single frame.

Timings: a track first seen at sample t_a and last seen at t_b, sampled every
P seconds, really began in (t_a - P, t_a] and ended in [t_b, t_b + P); the
estimate takes the middle of each interval, so the error is at most ±P/2.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .metrics import cer
from .srt import Cue
from .textmap import TextParams, detect_lines, pad_box

# A recogniser maps line crops (RGB uint8) to (text, confidence) pairs.
ReadFn = Callable[[Sequence[np.ndarray]], list[tuple[str, float]]]


@dataclass
class SubtitleParams:
    band: tuple[float, float] = (0.62, 1.0)     # vertical search band, fractions of the height
    centred: float = 0.30                       # |line centre - frame centre| <= this * width
    work_h: int = 360                           # detection runs on frames scaled to this height
    same_iou: float = 0.40                      # stroke IoU to continue a track
    box_iou: float = 0.50                       # box IoU to continue a track
    max_miss: int = 1                           # samples a track may go unseen (detection dropout)
    min_dur: float = 0.35                       # tracks shorter than this (s) are dropped
    max_dur: float = 10.0                       # longer tracks are static scene text, not subtitles
    max_median: int = 15                        # crops per track for the temporal median
    min_conf: float | None = None               # confidence below which a line is dropped (None: the reader's own)
    group_overlap: float = 0.6                  # shared time (of the shorter) for two lines to share a cue
    merge_gap: float = 0.6                      # same text across a gap <= this (s) -> one cue
    merge_cer: float = 0.3
    text: TextParams = field(default_factory=TextParams)


@dataclass(eq=False)
class LineTrack:
    box: tuple                  # current box, work coordinates of the band
    region: tuple               # fixed crop region, full-resolution band coordinates
    strokes: np.ndarray         # stroke map inside `box` at the last sample
    times: list = field(default_factory=list)
    crops: list = field(default_factory=list)
    text: str = ""
    conf: float = 0.0
    y: float = 0.0              # full-resolution centre row, for line order
    missed: int = 0             # consecutive samples without a match
    h: float = 0.0              # full-resolution line height

    @property
    def start(self):
        return self.times[0]

    @property
    def end(self):
        return self.times[-1]


def _resize(rgb: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1) < 1e-3:
        return rgb
    from PIL import Image

    h, w = rgb.shape[:2]
    return np.asarray(Image.fromarray(rgb).resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                                  Image.BILINEAR))


def box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def interval_iou(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


class SubtitleReader:
    def __init__(self, read_fn: ReadFn, params: SubtitleParams | None = None):
        self.read_fn = read_fn
        self.p = params or SubtitleParams()

    def detect(self, rgb: np.ndarray):
        """Line boxes of the subtitle band: (band, small band, scale, boxes, strokes)."""
        p = self.p
        h = rgb.shape[0]
        band = rgb[int(p.band[0] * h):int(p.band[1] * h)]
        scale = min(1.0, p.work_h / h)
        small = _resize(band, scale)
        boxes, strokes = detect_lines(small, p.text, frame_h=round(h * scale))
        cx = small.shape[1] / 2
        boxes = [b for b in boxes if abs((b[0] + b[2]) / 2 - cx) <= p.centred * small.shape[1]]
        return band, scale, boxes, strokes

    # ---------------------------------------------------------------- tracking
    def tracks(self, frames: Iterable[tuple[float, np.ndarray]]) -> Iterable[LineTrack]:
        """Yield each line track when it ends (so only live tracks hold crops)."""
        p = self.p
        active: list[LineTrack] = []
        for t, rgb in frames:
            band, scale, boxes, strokes = self.detect(rgb)
            nxt = []
            for b in boxes:
                x0, y0, x1, y1 = b
                patch = strokes[y0:y1, x0:x1]
                best, best_iou = None, 0.0
                for tr in active:
                    if tr in nxt or box_iou(b, tr.box) < p.box_iou:
                        continue
                    ux0, uy0 = min(x0, tr.box[0]), min(y0, tr.box[1])
                    ux1, uy1 = max(x1, tr.box[2]), max(y1, tr.box[3])
                    prev = np.zeros((uy1 - uy0, ux1 - ux0), bool)
                    bx0, by0, bx1, by1 = tr.box
                    prev[by0 - uy0:by1 - uy0, bx0 - ux0:bx1 - ux0] = tr.strokes
                    iou = mask_iou(prev, strokes[uy0:uy1, ux0:ux1])
                    if iou >= p.same_iou and iou > best_iou:
                        best, best_iou = tr, iou
                if best is None:
                    full = tuple(int(round(v / scale)) for v in b)
                    region = pad_box(full, band.shape[0], band.shape[1], fy=0.45, fx=0.8)
                    best = LineTrack(b, region, patch, y=(full[1] + full[3]) / 2, h=full[3] - full[1])
                best.box, best.strokes, best.missed = b, patch, 0
                best.times.append(t)
                rx0, ry0, rx1, ry1 = best.region
                best.crops.append(band[ry0:ry1, rx0:rx1].copy())
                nxt.append(best)
            for tr in active:
                if tr not in nxt:
                    tr.missed += 1
                    if tr.missed > p.max_miss:
                        yield tr
                    else:
                        nxt.append(tr)
            active = nxt
        yield from active

    def read_track(self, tr: LineTrack, scale: float) -> None:
        """Median of the track's crops -> re-find the line inside it -> text."""
        k = len(tr.crops)
        idx = range(k) if k <= self.p.max_median else np.linspace(0, k - 1, self.p.max_median).round().astype(int)
        med = np.median(np.stack([tr.crops[i] for i in idx]), axis=0).astype(np.uint8)
        rx0, ry0, _, _ = tr.region
        full = tuple(int(round(v / scale)) for v in tr.box)
        box = (full[0] - rx0, full[1] - ry0, full[2] - rx0, full[3] - ry0)
        x0, y0, x1, y1 = pad_box(box, med.shape[0], med.shape[1])
        (tr.text, tr.conf), = self.read_fn([med[y0:y1, x0:x1]])
        tr.text = tr.text.strip()
        tr.crops = []                                    # free memory

    # ---------------------------------------------------------------- whole clip
    def run(self, frames: Iterable[tuple[float, np.ndarray]], period: float) -> list[Cue]:
        """frames: (time, RGB) pairs sampled every `period` seconds."""
        p = self.p
        scale_box = [1.0]

        def watch():
            for t, rgb in frames:
                scale_box[0] = min(1.0, p.work_h / rgb.shape[0])
                yield t, rgb

        min_conf = reader_threshold(self.read_fn, "min_conf", 0.35) if p.min_conf is None else p.min_conf
        lines = []
        for tr in self.tracks(watch()):
            dur = tr.end - tr.start + period
            if dur < p.min_dur or dur > p.max_dur:
                continue
            self.read_track(tr, scale_box[0])
            if plausible(tr.text, tr.conf, min_conf):
                lines.append(tr)
        return merge_cues(group_lines(lines, period, p.group_overlap), p.merge_gap, p.merge_cer)


def reader_threshold(read_fn, name: str, default: float) -> float:
    """A recogniser's confidence scale is its own (ours: mean per-step
    probability; PP-OCR: its score), so each reader object may carry the
    threshold chosen for it on the tuning clips."""
    return float(getattr(getattr(read_fn, "__self__", None), name, default))


def plausible(text: str, conf: float, min_conf: float) -> bool:
    """Keep a reading? Texture read as text comes out short and unsure: "TAT",
    ".c.", "11". A line needs two letters or digits, and a short line needs
    more confidence than a long one."""
    n = sum(c.isalnum() for c in text)
    return n >= 2 and conf >= (min_conf if n >= 4 else min(0.95, min_conf + 0.3))


def group_lines(lines: list[LineTrack], period: float, min_overlap: float = 0.6) -> list[Cue]:
    """Lines that are on screen at the same time form one cue (top line first).

    A line joins a group when their time spans overlap by at least
    `min_overlap` of the shorter span. A second piece of a line the group
    already has (same row, similar text: the track broke for a few samples)
    is folded into it instead of becoming a duplicate line."""
    lines = sorted(lines, key=lambda tr: tr.start)
    groups: list[dict] = []           # {"start", "end", "rows": [LineTrack]}
    for tr in lines:
        t0, t1 = tr.start, tr.end + period
        for g in groups:
            inter = min(g["end"], t1) - max(g["start"], t0)
            if inter < min_overlap * min(g["end"] - g["start"], t1 - t0):
                continue
            same_row = [o for o in g["rows"] if abs(o.y - tr.y) < 0.6 * max(o.h, tr.h)]
            if same_row:
                o = same_row[0]
                if cer(o.text, tr.text) > 0.3 and cer(tr.text, o.text) > 0.3:
                    continue                  # different text in the same place: another subtitle
                if tr.conf > o.conf:
                    o.text, o.conf = tr.text, tr.conf
            else:
                g["rows"].append(tr)
            g["start"], g["end"] = min(g["start"], t0), max(g["end"], t1)
            break
        else:
            groups.append({"start": t0, "end": t1, "rows": [tr]})
    cues = []
    for g in groups:
        rows = sorted(g["rows"], key=lambda tr: tr.y)
        cues.append(Cue(max(0.0, g["start"] - period / 2), g["end"] - period / 2,
                        "\n".join(tr.text for tr in rows)))
    return sorted(cues, key=lambda c: c.start)


def merge_cues(cues: list[Cue], max_gap: float, max_cer: float) -> list[Cue]:
    """De-duplicate: the same text in consecutive cues separated by a short gap
    (a missed frame, a flash, a detection dropout) becomes one cue. Text that
    one cue only partly reads (a line missed) keeps the longer reading."""
    out: list[Cue] = []
    for c in cues:
        if out:
            prev = out[-1]
            a, b = prev.flat, c.flat
            if c.start - prev.end <= max_gap and (cer(a, b) <= max_cer or cer(b, a) <= max_cer
                                                  or (len(a) > 3 and a in b) or (len(b) > 3 and b in a)):
                prev.end = max(prev.end, c.end)
                if len(b) > len(a):
                    prev.text = c.text
                continue
        out.append(Cue(c.start, c.end, c.text))
    return out
