"""Character timeline for any clip: who is on screen when, and who is talking.

    python p3_tracking/timeline.py trailers/goldeneye.mp4
    python p3_tracking/timeline.py clip.mp4 --embedder sface --asd lightasd
    python p3_tracking/timeline.py clip.mp4 --detector centernet --det-ckpt p2_detection/checkpoints/centernet_r18.pt

-> outputs/<clip>/characters.json, outputs/<clip>/timeline.png

--embedder ours needs the checkpoint from train_embed.py
(p3_tracking/checkpoints/embed_triplet.pt); without it SFace is used.
--asd lightasd needs data/weights/light_asd_ava.model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cinescope.faces.characters import character_timeline, plot_timeline  # noqa: E402


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--detector", default="yunet", choices=["yunet", "centernet"])
    ap.add_argument("--det-ckpt")
    ap.add_argument("--embedder", default="ours", choices=["ours", "sface"])
    ap.add_argument("--embed-ckpt", default=str(ROOT / "p3_tracking" / "checkpoints" / "embed_triplet.pt"))
    ap.add_argument("--asd", default="ours", choices=["ours", "lightasd"])
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)
    if args.embedder == "ours" and not Path(args.embed_ckpt).exists():
        print(f"{args.embed_ckpt} not found (run train_embed.py); using SFace")
        args.embedder = "sface"
    out = Path(args.out) / Path(args.video).stem
    t0 = time.perf_counter()
    r = character_timeline(args.video, out, detector=args.detector, det_ckpt=args.det_ckpt,
                           embedder=args.embedder, embed_ckpt=args.embed_ckpt, asd=args.asd)
    r["seconds"] = round(time.perf_counter() - t0, 1)
    plot_timeline(r, out / "timeline.png")
    r.pop("_thumbs", None)
    (out / "characters.json").write_text(json.dumps(r, indent=1))
    print(f"{args.video}: {len(r['characters'])} characters from {r['n_tracks']} tracks "
          f"({r['seconds']} s) -> {out / 'characters.json'}, {out / 'timeline.png'}")
    for c in r["characters"][:10]:
        print(f"  {c['id']}: on screen {c['screen_time_s']} s, talking {c['speaking_time_s']} s, "
              f"{len(c['tracks'])} tracks")
    return r


if __name__ == "__main__":
    main()
