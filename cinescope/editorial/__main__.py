"""python -m cinescope.editorial outputs/<clip>/report.json [-f edl fcpxml otio csv vtt pdf png] [-o DIR]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .exports import FORMATS
from .pacing import pacing

SHEETS = ("pdf", "png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cinescope.editorial",
                                 description="Export a CineScope report for editing software, and print its pace.")
    ap.add_argument("report", help="outputs/<clip>/report.json (or its folder)")
    ap.add_argument("-f", "--formats", nargs="+", choices=[*FORMATS, *SHEETS], default=[*FORMATS, "pdf"],
                    help="what to write (default: every cut format, the subtitles and a PDF storyboard)")
    ap.add_argument("-o", "--out", help="folder for the files (default: exports/ next to the report)")
    a = ap.parse_args(argv)

    rp = Path(a.report)
    if rp.is_dir():
        rp = rp / "report.json"
    if not rp.is_file():
        print(f"no report at {rp}; run python -m cinescope.pipeline <video> first", file=sys.stderr)
        return 2
    rep = json.loads(rp.read_text(encoding="utf-8"))
    if not rep.get("shots"):
        print(f"{rp} has no shots (run the 'shots' stage)", file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else rp.parent / "exports"
    out.mkdir(parents=True, exist_ok=True)
    stem = rep.get("clip", rp.parent.name)
    for f in a.formats:
        if f in SHEETS:
            from .sheet import save_sheet

            for p in save_sheet(rep, rp.parent, out / f"{stem}_storyboard.{f}"):
                print(f"wrote {p}")
            continue
        fn, ext, _ = FORMATS[f]
        if f == "vtt" and not rep.get("subtitles"):
            print("skipped vtt: no subtitles in the report (run the 'subtitles' stage)")
            continue
        p = out / f"{stem}.{ext}"
        p.write_text(fn(rep), encoding="utf-8", newline="\n")
        print(f"wrote {p}")
    pc = pacing(rep)
    if pc:
        print(f"pace: {pc['n_shots']} shots, average {pc['asl_s']:.2f} s, median {pc['median_s']:.2f} s, "
              f"{pc['cuts_per_min']:.1f} cuts/min; fastest {pc['fastest']['cuts_per_min']:.0f} cuts/min "
              f"around {pc['fastest']['t']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
