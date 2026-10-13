"""Score subtitle readers on clips with burned-in subtitles of known text and timing.

    python p7_ocr/make_testset.py trailers/goldeneye.mp4 trailers/testvideo.mp4 trailers/Megamind.avi
    python p7_ocr/eval_subtitles.py --readers ours rapidocr opencv rapidocr-det
    # -> p7_ocr/results/subtitles_test.json and .md

Each video `<clip>.<style>.mp4` in --videos is scored against
`<labels>/<clip>.<style>.srt`. Two CERs: exact (case, punctuation and spaces
count) and folded (lower-case letters and digits only — the only comparison
the OpenCV zoo CRNN, which has no capitals or punctuation, can take part in).
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

from read_subtitles import read_clip  # noqa: E402

from cinescope.ocr.charset import fold  # noqa: E402
from cinescope.ocr.srt import read_srt, score, write_srt  # noqa: E402
from cinescope.ocr.subtitles import SubtitleParams  # noqa: E402


def pooled(rows):
    """Corpus-level numbers over several videos (sums, not means of ratios)."""
    keys = ("cues", "predicted", "matched", "chars", "edits", "missed_chars", "false_chars")
    tot = {k: sum(r[k] for r in rows) for k in keys}
    tot["cer"] = (tot["edits"] + tot["missed_chars"] + tot["false_chars"]) / max(1, tot["chars"])
    tot["cer_matched"] = tot["edits"] / max(1, sum(r["chars"] - r["missed_chars"] for r in rows))
    tot["recall"] = tot["matched"] / max(1, tot["cues"])
    tot["precision"] = tot["matched"] / max(1, tot["predicted"])
    m = max(1, tot["matched"])
    tot["start_err_s"] = sum(r["start_err_s"] * r["matched"] for r in rows) / m
    tot["end_err_s"] = sum(r["end_err_s"] * r["matched"] for r in rows) / m
    tot["exact_cues"] = sum(r["exact_cues"] * r["cues"] for r in rows) / max(1, tot["cues"])
    return tot


def table(results: dict, styles: list[str]) -> str:
    lines = ["| Reader | CER | CER folded | CER on found cues | cue recall | cue precision | exact cues "
             "| start err (s) | end err (s) | " + " | ".join(f"CER {s}" for s in styles) + " | time (s) |",
             "|---" * (10 + len(styles)) + "|"]
    for reader, r in results.items():
        a, f = r["all"], r["all_folded"]
        by = " | ".join(f"{r['by_style'][s]['cer']:.3f}" if s in r["by_style"] else "—" for s in styles)
        lines.append(f"| {reader} | {a['cer']:.3f} | {f['cer']:.3f} | {a['cer_matched']:.3f} "
                     f"| {a['recall']:.3f} | {a['precision']:.3f} | {a['exact_cues']:.3f} "
                     f"| {a['start_err_s']:.3f} | {a['end_err_s']:.3f} | {by} | {r['seconds']:.0f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=str(ROOT / "outputs" / "p7" / "burned"))
    ap.add_argument("--labels", default=str(ROOT / "p7_ocr" / "labels" / "burned"))
    ap.add_argument("--readers", nargs="+", default=["ours", "rapidocr", "opencv", "rapidocr-det"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--decoder", default="greedy")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--min-conf", type=float, default=None, help="override SubtitleParams.min_conf")
    ap.add_argument("--name", default="test")
    ap.add_argument("--tag", default=None, help="row name in the results (default: the reader)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these <clip>.<style> keys")
    args = ap.parse_args(argv)

    videos = sorted(Path(args.videos).glob("*.mp4"))
    if args.only:
        videos = [v for v in videos if v.stem in args.only]
    pred_dir = ROOT / "outputs" / "p7" / f"pred_{args.name}"
    out = ROOT / "p7_ocr" / "results" / f"subtitles_{args.name}.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    styles = sorted({v.stem.split(".")[-1] for v in videos})
    for reader in args.readers:
        tag = args.tag or reader
        rows, rows_fold = {}, {}
        t0 = time.time()
        for v in videos:
            truth = read_srt(Path(args.labels) / f"{v.stem}.srt")
            params = SubtitleParams() if args.min_conf is None else SubtitleParams(min_conf=args.min_conf)
            cues = read_clip(str(v), reader, args.fps, args.ckpt, args.decoder, params)
            (pred_dir / tag).mkdir(parents=True, exist_ok=True)
            write_srt(cues, pred_dir / tag / f"{v.stem}.srt")
            rows[v.stem] = score(truth, cues)
            rows_fold[v.stem] = score(truth, cues, norm=fold)
            r = rows[v.stem]
            print(f"{tag:12s} {v.stem:24s} CER {r['cer']:.3f} folded {rows_fold[v.stem]['cer']:.3f} "
                  f"recall {r['recall']:.2f} precision {r['precision']:.2f} "
                  f"start {r['start_err_s']:.3f}s end {r['end_err_s']:.3f}s", flush=True)
        results[tag] = {
            "all": pooled(list(rows.values())), "all_folded": pooled(list(rows_fold.values())),
            "by_style": {s: pooled([r for k, r in rows.items() if k.endswith("." + s)]) for s in styles},
            "by_video": rows, "seconds": time.time() - t0,
            "settings": {"reader": reader, "fps": args.fps, "decoder": args.decoder, "ckpt": args.ckpt,
                         "min_conf": args.min_conf},
        }
        out.write_text(json.dumps(results, indent=1))
    md = table(results, styles)
    out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
