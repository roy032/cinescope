"""Scene search in plain English.

    python p8_search/search.py trailers/goldeneye.mp4 "the scene where a man walks away from an explosion"
    python p8_search/search.py clip.mp4 "a plane in fog" --k 3 --index ivfpq
    python p8_search/search.py clip.mp4 "a red car" --encoder scratch:p8_search/checkpoints/dual_frames.pt

The first query on a clip builds its index (Phase 0 shots -> key frames -> embeddings)
in outputs/<clip>/index/; later queries only embed the text. Prints the top-k shots
with timestamps and writes a contact sheet to outputs/<clip>/search/<query>.jpg.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cinescope.search.encoders import get_encoder  # noqa: E402
from cinescope.search.rag import TAG_VOCAB, fmt_t  # noqa: E402
from cinescope.search.shotindex import ShotIndex, build_clip_index, index_dir  # noqa: E402
from cinescope.search.vecindex import IndexSpec  # noqa: E402


def open_index(video: str, encoder, out: str = "outputs", rebuild: bool = False) -> ShotIndex:
    d = index_dir(out, video, encoder.name)
    if not rebuild and (d / "docs.json").exists():
        ix = ShotIndex.load(d)
        if ix.encoder_name == encoder.name:
            return ix
    return build_clip_index(video, encoder, d, tag_vocab=TAG_VOCAB)


def result_sheet(ix: ShotIndex, hits: list[dict], query: str, path: Path, thumb_h: int = 120) -> None:
    rows = []
    for h in hits:
        thumbs = []
        for j in range(len(h.get("frames", []))):
            p = ix.frame_path(h, j)
            if p and p.exists():
                im = Image.open(p).convert("RGB")
                thumbs.append(np.asarray(im.resize((max(1, round(im.width * thumb_h / im.height)), thumb_h))))
        strip = np.concatenate(thumbs, 1) if thumbs else np.zeros((thumb_h, thumb_h, 3), np.uint8)
        label = Image.new("RGB", (300, thumb_h), (18, 18, 18))
        d = ImageDraw.Draw(label)
        d.text((8, 6), f"#{h['rank']}  score {h['score']:.3f}", fill=(240, 200, 120))
        d.text((8, 24), f"{h.get('clip', '')} shot {h['shot']}" +
               (f" ({h['part'] + 1}/{h['n_parts']})" if h.get("n_parts", 1) > 1 else ""), fill=(230, 230, 230))
        d.text((8, 42), f"{fmt_t(h['t0'])} - {fmt_t(h['t1'])}", fill=(140, 220, 160))
        for i, t in enumerate(h.get("tags", [])[:3]):
            d.text((8, 62 + 16 * i), t["tag"][:40], fill=(150, 150, 150))
        rows.append(np.concatenate([np.asarray(label), strip], 1))
    if not rows:
        return
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in rows]
    head = Image.new("RGB", (w, 24), (0, 0, 0))
    ImageDraw.Draw(head).text((8, 6), f'"{query}"', fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([np.asarray(head)] + rows, 0)).save(path, quality=85)


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--encoder", default="openclip", help="openclip | openclip:<model>:<pretrained> | scratch:<ckpt>")
    ap.add_argument("--index", default="flat", choices=["flat", "ivf", "ivfpq"],
                    help="vector index over the segments (flat = exact; the others matter at film scale)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)

    enc = get_encoder(args.encoder)
    ix = open_index(args.video, enc, args.out, args.rebuild)
    t = time.perf_counter()
    hits = ix.search_text(args.query, enc, k=args.k, spec=IndexSpec(args.index))
    ms = (time.perf_counter() - t) * 1000
    print(f'"{args.query}"  ({len(ix)} segments, {enc.name}, {args.index}, {ms:.0f} ms)')
    for h in hits:
        tags = ", ".join(x["tag"] for x in h.get("tags", [])[:3])
        part = "" if h.get("n_parts", 1) == 1 else f" ({h['part'] + 1}/{h['n_parts']})"
        print(f"  #{h['rank']}  {fmt_t(h['t0'])}–{fmt_t(h['t1'])}  shot {h['shot']:3d}{part}"
              f"  score {h['score']:.3f}  [{tags}]")
    slug = re.sub(r"[^a-z0-9]+", "_", args.query.lower()).strip("_")[:60] or "query"
    sheet = Path(args.out) / Path(args.video).stem / "search" / f"{slug}.jpg"
    result_sheet(ix, hits, args.query, sheet)
    print(f"-> {sheet}")
    return hits


if __name__ == "__main__":
    main()
