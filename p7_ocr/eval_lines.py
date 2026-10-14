"""Recognition alone: line-level CER on rendered subtitle crops.

    python p7_ocr/eval_lines.py --bg-clips trailers/goldeneye.mp4 trailers/testvideo.mp4 trailers/Megamind.avi
    # -> p7_ocr/results/lines_test.json

The crops are rendered like the training lines, but with test text (Sherlock
Holmes lines), held-out fonts (DejaVu, never seen in training) and frames of
the held-out test clips as backgrounds. No detection, no tracking: this
isolates the recogniser from the rest of the subtitle pipeline.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from cinescope.ocr.charset import fold  # noqa: E402
from cinescope.ocr.libs import make_reader  # noqa: E402
from cinescope.ocr.metrics import corpus_cer, corpus_wer  # noqa: E402
from cinescope.ocr.synth import find_fonts, frames_from_clips, render_line  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bg-clips", nargs="+", required=True)
    ap.add_argument("--lines", default=str(ROOT / "p7_ocr" / "labels" / "sherlock_lines.txt"))
    ap.add_argument("--fonts", nargs="+", default=["DejaVu"], help="font-file name parts to use")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--readers", nargs="+", default=["ours", "ours-beam", "rapidocr", "opencv"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--name", default="test")
    args = ap.parse_args(argv)

    rng = random.Random(2024)
    texts = [s.strip() for s in Path(args.lines).read_text().splitlines() if s.strip()][:args.n]
    fonts = [f for f in find_fonts() if any(p.lower() in Path(f).name.lower() for p in args.fonts)]
    frames = frames_from_clips(args.bg_clips, per_clip=30)
    # crops are rendered at 64 px (the recogniser resizes to 32 itself), RGB
    # is not needed: every reader gets the same grey crop as a 3-channel image
    crops = [np.asarray(render_line(t, rng, fonts, frames, out_h=64).convert("RGB")) for t in texts]
    print(f"{len(crops)} crops, fonts: {[Path(f).name for f in fonts]}")
    out = ROOT / "p7_ocr" / "results" / f"lines_{args.name}.json"
    results = {}
    for name in args.readers:
        reader = make_reader("ours", args.ckpt, "beam") if name == "ours-beam" else make_reader(name, args.ckpt)
        t0 = time.time()
        hyps = [t for t, _ in reader.read(crops)]
        dt = (time.time() - t0) / len(crops) * 1000
        r = {"cer": corpus_cer(texts, hyps), "wer": corpus_wer(texts, hyps),
             "cer_folded": corpus_cer([fold(t) for t in texts], [fold(h) for h in hyps]),
             "exact": float(np.mean([a == b for a, b in zip(texts, hyps, strict=True)])),
             "ms_per_line": dt, "examples": list(zip(texts[:8], hyps[:8], strict=True))}
        results[name] = r
        print(f"{name:10s} CER {r['cer']:.4f}  folded {r['cer_folded']:.4f}  WER {r['wer']:.4f}  "
              f"exact {r['exact']:.3f}  {dt:.1f} ms/line", flush=True)
    out.write_text(json.dumps({"n": len(crops), "fonts": [Path(f).name for f in fonts],
                               "bg_clips": [Path(c).name for c in args.bg_clips], "readers": results}, indent=1))


if __name__ == "__main__":
    main()
