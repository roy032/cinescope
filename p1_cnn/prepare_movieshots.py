"""Flatten MovieShots annotations into data/movieshots/index.csv.

    python p1_cnn/prepare_movieshots.py --root D:/datasets/MovieShots
    python p1_cnn/prepare_movieshots.py --root ... --annotations v1_split_trailer.json

MovieShots is distributed by its authors on request (see p1_cnn/README.md);
it is never committed here. The script reports how many shots it could not
use and why, so a layout mismatch shows up as numbers, not as a silent drop.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.framing.data import assign_splits, read_movieshots, write_index  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="MovieShots folder (with trailer/ and the JSON files)")
    ap.add_argument("--annotations", help="annotation JSON (default: *split*.json in --root)")
    ap.add_argument("--out", default="data/movieshots/index.csv")
    args = ap.parse_args(argv)

    records, problems = read_movieshots(args.root, args.annotations)
    if not records:
        sys.exit(f"no usable shots found; problems: {dict(problems)}. "
                 "Check --root and the layout described in cinescope/framing/data.py")
    if {r.split for r in records} <= {"all"}:
        assign_splits(records)
        print("annotations had no splits: split 70/10/20 by trailer")
    write_index(records, args.out)
    by_split = Counter(r.split for r in records)
    print(f"wrote {len(records)} shots to {args.out}: {dict(by_split)}")
    for split in sorted(by_split):
        c = Counter(r.scale for r in records if r.split == split)
        print(f"  {split:<5} " + "  ".join(f"{k} {v}" for k, v in sorted(c.items())))
    if problems:
        print(f"skipped: {dict(problems)}")


if __name__ == "__main__":
    main()
