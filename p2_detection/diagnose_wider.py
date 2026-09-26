"""Break WIDER FACE val results down by face size (after eval_wider.py).

    python p2_detection/diagnose_wider.py --wider D:/datasets/WIDER \
        --preds p2_detection/results/centernet_r18/wider_val_preds.npz --eval-tools .../ground_truth

Prints, per size bucket, how many faces are found, mislocalised, missed or
found only at a low score, and how many duplicates they attract — for all
faces and for the official Easy subset. Scores are min-max normalised over the
whole set first, exactly as the WIDER evaluation does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.detect.diagnose import breakdown, markdown  # noqa: E402
from cinescope.detect.wider import parse_annotations  # noqa: E402


def load_preds(path: str | Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    preds = {k: z[f"p{i}"] for i, k in enumerate(keys)}
    scores = np.concatenate([p[:, 4] for p in preds.values() if len(p)] or [np.zeros(1)])
    lo, span = scores.min(), max(scores.max() - scores.min(), 1e-12)
    for p in preds.values():
        if len(p):
            p[:, 4] = (p[:, 4] - lo) / span
    return preds


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wider", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--eval-tools", help="official ground_truth folder, for the Easy/Medium/Hard subsets")
    ap.add_argument("--out", help="write the tables here (markdown)")
    args = ap.parse_args(argv)

    preds = load_preds(args.preds)
    if args.eval_tools:
        from cinescope.detect.wider_eval import load_ground_truth

        gts, keeps = load_ground_truth(args.eval_tools)
    else:
        items = parse_annotations(Path(args.wider) / "wider_face_split" / "wider_face_val_bbx_gt.txt")
        gts, keeps = {it.path[:-4]: it.boxes for it in items}, {}
    report = {"all faces": breakdown(gts, preds)}
    for s in ("easy", "medium", "hard"):
        if s in keeps:
            report[s] = breakdown(gts, preds, keeps[s])
    text = "\n\n".join(f"### {name}\n\n{markdown(rows)}" for name, rows in report.items())
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
