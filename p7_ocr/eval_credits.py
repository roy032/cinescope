"""Score the credits / on-screen text reader against known lines.

    python p7_ocr/make_credits_clip.py                     # synthetic credits, known text
    python p7_ocr/eval_credits.py p7_ocr/labels/credits_test.json --readers ours rapidocr opencv
    python p7_ocr/eval_credits.py p7_ocr/labels/goldeneye_text.json --video trailers/goldeneye.mp4

A label file names the video and lists the text lines on screen. A true line
counts as found when some output line reads it with CER <= 0.2 (one-to-one
matching); precision is found / output lines, so reading texture as text costs.
The OpenCV zoo CRNN is always compared folded (lower-case letters and digits).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "p7_ocr"))

from read_subtitles import frames_of  # noqa: E402

from cinescope.ocr.charset import fold  # noqa: E402
from cinescope.ocr.credits import CreditsParams, CreditsReader, score_lines  # noqa: E402
from cinescope.ocr.libs import make_reader  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels")
    ap.add_argument("--video", default=None, help="override the label file's video path")
    ap.add_argument("--readers", nargs="+", default=["ours", "rapidocr", "opencv"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--fold", action="store_true", help="compare lower-case alphanumerics only")
    ap.add_argument("--min-conf", type=float, default=None, help="override the reader's credits threshold")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    lab = json.loads(Path(args.labels).read_text())
    video = args.video or str(ROOT / lab["video"])
    truth = lab["lines"]
    out = ROOT / "p7_ocr" / "results" / f"credits_{Path(args.labels).stem}.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    for reader in args.readers:
        rec = make_reader(reader, args.ckpt)
        t0 = time.time()
        lines = CreditsReader(rec.read, CreditsParams(min_conf=args.min_conf)).run(frames_of(video, args.fps))
        texts = [ln["text"] for ln in lines]
        folded = args.fold or reader == "opencv"
        norm = fold if folded else (lambda s: s)
        r = score_lines([norm(t) for t in truth], [norm(t) for t in texts])
        r.update(folded=folded, seconds=time.time() - t0, output=texts)
        if not folded:                               # also report the folded score
            rf = score_lines([fold(t) for t in truth], [fold(t) for t in texts])
            r["folded_recall"], r["folded_mean_cer"] = rf["recall"], rf["mean_cer"]
        results[args.tag or reader] = r
        print(f"{args.tag or reader:10s} found {r['found']}/{r['lines']}  recall {r['recall']:.3f}  "
              f"precision {r['precision']:.3f}  mean CER {r['mean_cer']:.3f}  ({r['seconds']:.0f} s)",
              flush=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
