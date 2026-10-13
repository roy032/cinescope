"""Read the burned-in subtitles (and on-screen text / credits) of any clip.

    python p7_ocr/read_subtitles.py trailers/goldeneye.mp4
    # -> outputs/goldeneye/subtitles.srt, credits.txt

    --reader ours        our CRNN (p7_ocr/checkpoints/crnn.pt; --ckpt to change)
             rapidocr    PP-OCRv3 recogniser on our detected lines
             opencv      OpenCV zoo CRNN on our detected lines (lower-case words only)
             trocr       TrOCR, if transformers and local weights are available
             rapidocr-det  PP-OCRv3 detector + recogniser on every sampled frame
                           (library end to end; our temporal grouping only)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cinescope.io.video import VideoReader  # noqa: E402
from cinescope.ocr.credits import CreditsReader  # noqa: E402
from cinescope.ocr.libs import RapidOCRFull, make_reader  # noqa: E402
from cinescope.ocr.metrics import cer  # noqa: E402
from cinescope.ocr.srt import Cue, write_srt  # noqa: E402
from cinescope.ocr.subtitles import SubtitleParams, SubtitleReader, merge_cues  # noqa: E402


def frames_of(clip: str, fps: float, start: float = 0.0, end: float | None = None):
    with VideoReader(clip) as vr:
        for f in vr.frames(sample_fps=fps, start_s=start, end_s=end):
            yield f.t, f.rgb


def library_cues(clip: str, fps: float, params: SubtitleParams, min_score: float = 0.5,
                 same_cer: float = 0.3) -> list[Cue]:
    """Baseline that uses the library's own detector in the subtitle band on
    every sample; consecutive samples with near-identical text form a cue."""
    eng = RapidOCRFull()
    period = 1.0 / fps
    runs: list[list] = []                        # [t_first, t_last, [texts]]
    prev_text = None
    for t, rgb in frames_of(clip, fps):
        h, w = rgb.shape[:2]
        y0 = int(params.band[0] * h)
        res = [r for r in eng.detect_read(rgb[y0:]) if r[2] >= min_score
               and abs((r[0][0] + r[0][2]) / 2 - w / 2) <= params.centred * w]
        # lines: group words whose boxes share a row, left to right
        res.sort(key=lambda r: (r[0][1] + r[0][3]) / 2)
        lines: list[list] = []
        for box, text, _ in res:
            yc = (box[1] + box[3]) / 2
            if lines and abs(yc - lines[-1][0]) < 0.5 * (box[3] - box[1]):
                lines[-1][1].append((box[0], text))
            else:
                lines.append([yc, [(box[0], text)]])
        text = "\n".join(" ".join(s for _, s in sorted(ws)) for _, ws in lines)
        if not text:
            prev_text = None
            continue
        if prev_text is not None and runs and cer(prev_text, text) <= same_cer:
            runs[-1][1] = t
            runs[-1][2].append(text)
        else:
            runs.append([t, t, [text]])
        prev_text = text
    cues = []
    for a, b, texts in runs:
        if b - a + period < params.min_dur:
            continue
        best = max(set(texts), key=texts.count)
        cues.append(Cue(max(0.0, a - period / 2), b + period / 2, best))
    return merge_cues(cues, params.merge_gap, params.merge_cer)


def read_clip(clip: str, reader: str = "ours", fps: float = 8.0, ckpt: str | None = None,
              decoder: str = "greedy", params: SubtitleParams | None = None) -> list[Cue]:
    params = params or SubtitleParams()
    if reader == "rapidocr-det":
        return library_cues(clip, fps, params)
    rec = make_reader(reader, ckpt, decoder)
    return SubtitleReader(rec.read, params).run(frames_of(clip, fps), 1.0 / fps)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--reader", default="ours",
                    choices=["ours", "rapidocr", "opencv", "trocr", "rapidocr-det"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--decoder", default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--fps", type=float, default=8.0, help="samples per second for subtitles")
    ap.add_argument("--min-conf", type=float, default=None, help="drop lines read with less confidence")
    ap.add_argument("--credits-fps", type=float, default=4.0)
    ap.add_argument("--no-credits", action="store_true")
    ap.add_argument("--out-dir", default=None, help="default outputs/<clip name>/")
    args = ap.parse_args(argv)

    out = Path(args.out_dir or ROOT / "outputs" / Path(args.clip).stem)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    params = SubtitleParams() if args.min_conf is None else SubtitleParams(min_conf=args.min_conf)
    cues = read_clip(args.clip, args.reader, args.fps, args.ckpt, args.decoder, params)
    write_srt(cues, out / "subtitles.srt")
    print(f"{len(cues)} cues -> {out / 'subtitles.srt'} ({time.time() - t0:.1f} s)")
    if not args.no_credits:
        t0 = time.time()
        reader = "ours" if args.reader == "rapidocr-det" else args.reader
        rec = make_reader(reader, args.ckpt, args.decoder)
        lines = CreditsReader(rec.read).run(frames_of(args.clip, args.credits_fps))
        subs = [ln for c in cues for ln in [c.flat, *c.text.split("\n")]]
        # subtitles are on-screen text too; credits.txt lists everything else
        lines = [ln for ln in lines if not any(cer(s, ln["text"]) < 0.3 for s in subs)]
        with open(out / "credits.txt", "w", encoding="utf-8") as f:
            for ln in lines:
                f.write(f"{ln['t']:7.2f}  {ln['text']}\n")
        (out / "credits.json").write_text(json.dumps(lines, indent=1))
        print(f"{len(lines)} on-screen text lines -> {out / 'credits.txt'} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
