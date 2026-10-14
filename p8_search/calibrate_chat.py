"""Choose the chat's "no matching segment" thresholds on the development clip only.

    python p8_search/calibrate_chat.py trailers/vtest.avi

Positives are the dev queries in labels/queries.json (things that are in the clip);
negatives are things that are not in it (written for vtest's surveillance view of a
lawn and path). For each query: the best CLIP similarity and the contrast (best minus
the median over the clip's segments). Prints both distributions and the thresholds
halfway between the weakest positive and the strongest negative; RagParams uses them.
The test clips are never looked at here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "p8_search"))

from search import open_index  # noqa: E402

from cinescope.search.encoders import get_encoder  # noqa: E402

NEGATIVES = ["a car chase", "an explosion", "a plane flying in the sky", "a dog running", "people dancing at a party",
             "a snowy mountain", "a man swimming in the sea", "a computer screen", "a cartoon character",
             "a deer in a forest", "a close-up of a woman's face", "a candlelit dinner"]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--labels", default=str(ROOT / "p8_search" / "labels" / "queries.json"))
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)
    lab = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    stem = Path(args.video).stem
    assert lab["clips"][stem]["split"] == "dev", "calibrate on the dev clip only"
    pos = [q["q"] for q in lab["clips"][stem]["queries"]]
    enc = get_encoder("openclip")
    ix = open_index(args.video, enc, args.out)
    rows = {}
    for kind, qs in (("positive", pos), ("negative", NEGATIVES)):
        s = enc.encode_texts(qs) @ ix.vectors.T
        best, contrast = s.max(1), s.max(1) - np.median(s, 1)
        rows[kind] = [{"q": q, "best": round(float(b), 4), "contrast": round(float(c), 4)}
                      for q, b, c in zip(qs, best, contrast, strict=True)]
    res = {"clip": stem, **rows}
    for key in ("best", "contrast"):
        lo_pos = min(r[key] for r in rows["positive"])
        hi_neg = max(r[key] for r in rows["negative"])
        res[f"{key}_threshold"] = round((lo_pos + hi_neg) / 2, 4)
        res[f"{key}_separates"] = lo_pos > hi_neg
        print(f"{key:9s} weakest positive {lo_pos:.4f}  strongest negative {hi_neg:.4f}  "
              f"-> threshold {res[f'{key}_threshold']:.4f} ({'separates' if lo_pos > hi_neg else 'overlaps'})")
    for kind in ("positive", "negative"):
        for r in rows[kind]:
            print(f"  {kind:8s} best {r['best']:.3f}  contrast {r['contrast']:+.3f}  {r['q']}")
    out = ROOT / "p8_search" / "results" / "chat_calibration.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"-> {out}")
    return res


if __name__ == "__main__":
    main()
