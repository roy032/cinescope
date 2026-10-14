"""Burn known subtitles into any clip: a subtitle test set from any video.

    python p7_ocr/make_testset.py trailers/goldeneye.mp4 --lines p7_ocr/labels/sherlock_lines.txt
    # -> p7_ocr/labels/burned/goldeneye.<style>.srt   (the truth, small, committed)
    #    outputs/p7/burned/goldeneye.<style>.mp4      (the clip with burned-in subtitles)

The text is public-domain dialogue (Sherlock Holmes for the test set, the
held-out end of Pride and Prejudice for the tuning set); timings follow
reading speed (see cinescope.ocr.srt.schedule). FFmpeg's libass renders the
subtitles exactly as a player or a fansub encoder would, in three styles:

  white   Arial (Liberation Sans on Linux), white, thin black outline — the default look
  yellow  DejaVu Sans Bold, yellow, drop shadow — a font the recogniser never trained on
  boxed   Liberation Serif in a translucent black box (BorderStyle 3)

Each clip/style pair reads the lines file from its own (hash-seeded) offset,
so different clips and styles mostly show different sentences.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cinescope.io.video import VideoReader  # noqa: E402
from cinescope.ocr.srt import schedule, write_srt  # noqa: E402

STYLES = {
    "white": "FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
             "BorderStyle=1,Outline=1.2,Shadow=0.5,MarginV=18",
    "yellow": "FontName=DejaVu Sans,Bold=1,FontSize=16,PrimaryColour=&H0000FFFF,"
              "OutlineColour=&H00000000,BorderStyle=1,Outline=0.6,Shadow=1.5,MarginV=14",
    "boxed": "FontName=Liberation Serif,FontSize=18,PrimaryColour=&H00FFFFFF,BackColour=&H60000000,"
             "OutlineColour=&H60000000,BorderStyle=3,Outline=2,Shadow=0,MarginV=22",
}


def burn(clip: str, srt_path: Path, style: str, out: Path, crf: int = 23) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(srt_path, Path(tmp) / "subs.srt")       # avoid escaping paths inside the filter
        vf = f"subtitles=subs.srt:force_style='{STYLES[style]}'"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(Path(clip).resolve()), "-vf", vf,
               "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast", "-an", str(out.resolve())]
        subprocess.run(cmd, check=True, cwd=tmp)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--lines", default=str(ROOT / "p7_ocr" / "labels" / "sherlock_lines.txt"))
    ap.add_argument("--styles", nargs="+", default=list(STYLES))
    ap.add_argument("--labels-dir", default=str(ROOT / "p7_ocr" / "labels" / "burned"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "p7" / "burned"))
    args = ap.parse_args(argv)

    lines = [s.strip() for s in Path(args.lines).read_text(encoding="utf-8").splitlines() if s.strip()]
    for clip in args.clips:
        name = Path(clip).stem
        with VideoReader(clip) as vr:
            dur = vr.info.duration_s
        for style in args.styles:
            key = f"{name}.{style}"
            seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            offset = seed % len(lines)
            ordered = lines[offset:] + lines[:offset]
            cues = schedule(ordered, dur, rng)
            srt_path = Path(args.labels_dir) / f"{key}.srt"
            srt_path.parent.mkdir(parents=True, exist_ok=True)
            write_srt(cues, srt_path)
            out = Path(args.out_dir) / f"{key}.mp4"
            burn(clip, srt_path, style, out)
            print(f"{key}: {len(cues)} cues -> {out}")


if __name__ == "__main__":
    main()
