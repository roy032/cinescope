"""Make sheets for labelling who speaks when (ground truth for Week 11).

    pip install silero-vad
    python p3_tracking/label_speech.py trailers/goldeneye.mp4

For every face track, one row per second: the lower half of the aligned face
at 12 frames per second, large enough to see the mouth open and close, with a
bar underneath showing where Silero VAD hears speech (green; without
silero-vad, a 0-4 kHz spectrogram instead) and a time ruler.

Labelling rule: a face is *speaking* where speech is audible **and** its
mouth visibly articulates (opens and closes in changing shapes). A face
with its mouth still while speech is heard is a listener (the speaker is off
screen or another face). Laughing, chewing and gasping are not speaking.
The labeller writes speaking intervals per *identity* (not per track) into
p3_tracking/labels/<clip>.speech.json:

    {"intervals": [{"id": "boris", "t0": 51.2, "t1": 53.0}, ...]}

The sheets show no output of the systems being compared (our active-speaker
score, Light-ASD). Silero is used only to say *whether* speech is audible --
a listener cannot hear the sheet -- so neither compared system sees it.
-> outputs/speech_<clip>/sheet_XX.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.speaker.audio import SR, load_audio, silero_vad  # noqa: E402
from p3_tracking.label_sheets import font  # noqa: E402
from p3_tracking.train_embed import clip_crops  # noqa: E402


def spectrogram(x: np.ndarray, sr: int = SR, win: int = 512, hop: int = 160, fmax: float = 4000) -> np.ndarray:
    if len(x) < win:
        return np.zeros((64, 1))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None] + hop * np.arange(n)[:, None]
    s = np.abs(np.fft.rfft(x[idx] * np.hanning(win), axis=1)) ** 2
    f = np.fft.rfftfreq(win, 1 / sr)
    return 10 * np.log10(s[:, f <= fmax].T + 1e-10)[::-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--min-frames", type=int, default=12)
    ap.add_argument("--rows-per-sheet", type=int, default=14)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)
    stem = Path(args.video).stem
    crops = clip_crops(args.video, Path(args.out), 24)
    from p3_tracking.track_clip import load_or_detect

    times = load_or_detect(args.video, Path(args.out) / stem).times
    fps = 1 / np.median(np.diff(times))
    audio = load_audio(args.video)
    sil = silero_vad(audio)                         # 100 per second, or None
    lo = np.percentile(spectrogram(audio[: SR * 60]), 5)
    step = max(1, int(round(fps / 12)))
    cw, ch, left, bar = 128, 64, 92, 26
    f = font(14)
    rows = []
    for c in crops:
        if len(c.track) < args.min_frames:
            continue
        t = times[np.asarray(c.track.frames)]
        for r0 in np.arange(np.floor(t[0] * 2) / 2, t[-1] + 1e-6, 1.0):
            r1 = r0 + 1.0
            row = Image.new("RGB", (left + 12 * cw, ch + bar + 16), (20, 20, 20))
            d = ImageDraw.Draw(row)
            d.text((4, 4), f"trk {c.track.id}", fill=(255, 255, 0), font=f)
            d.text((4, 24), f"{r0:.1f}s", fill=(255, 255, 255), font=f)
            for i in np.flatnonzero((t >= r0) & (t < r1))[::step]:
                x = left + min(11, int((t[i] - r0) * 12)) * cw
                row.paste(Image.fromarray(c.gray[i][56:112]).resize((cw, ch)).convert("RGB"), (x, 0))
            if sil is not None:
                p = sil[int(r0 * 100):int(r1 * 100)]
                for k, v in enumerate(p):
                    x = left + int(k / 100 * 12 * cw)
                    col = (40, 220, 60) if v > 0.5 else (60, 60, 60)
                    d.rectangle([x, ch + 4, x + int(12 * cw / 100), ch + bar - 2], fill=col)
            else:
                sp = spectrogram(audio[int(r0 * SR):int(r1 * SR)])
                g = np.clip((sp - lo) / 60.0, 0, 1)
                row.paste(Image.fromarray((g * 255).astype(np.uint8)).resize((12 * cw, bar)).convert("RGB"),
                          (left, ch + 2))
            for k in range(5):                                   # ruler every 0.25 s
                x = left + int(k * 0.25 * 12 * cw)
                d.line([(x, ch + bar), (x, ch + bar + 14)], fill=(255, 80, 80))
                d.text((x + 3, ch + bar), f"{r0 + k * 0.25:.2f}", fill=(255, 120, 120), font=f)
            rows.append(row)
    out = Path(args.out) / f"speech_{stem}"
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("sheet_*.png"):
        p.unlink()
    for s in range(0, len(rows), args.rows_per_sheet):
        group = rows[s:s + args.rows_per_sheet]
        sheet = Image.new("RGB", (group[0].width, sum(r.height + 4 for r in group)), (90, 90, 90))
        y = 0
        for r in group:
            sheet.paste(r, (0, y))
            y += r.height + 4
        sheet.save(out / f"sheet_{s // args.rows_per_sheet:02d}.png")
    print(f"{len(rows)} rows on {(len(rows) + args.rows_per_sheet - 1) // args.rows_per_sheet} sheets in {out}")


if __name__ == "__main__":
    main()
