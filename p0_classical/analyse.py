"""Phase 0 demo: split a clip into shots, label camera moves, extract palettes.

    python p0_classical/analyse.py trailers/Ted_Lasso_120s.mkv
    python p0_classical/analyse.py clip.mp4 --no-camera --out outputs

Writes to outputs/<clip>/:
  shots.json      every transition and shot, with palette and camera move
  barcode.png     the "movie barcode": one line per second, mean colour
  shots.jpg       one row per shot: middle frame, palette strip, camera label
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.io.video import VideoReader  # noqa: E402
from cinescope.palette.kmeans import Palette, palette_strip  # noqa: E402
from cinescope.shots.pipeline import analyse  # noqa: E402


def shot_sheet(video: str, result, out: Path, width: int = 192, max_rows: int = 400) -> None:
    """Middle frame of each shot + its palette + label, stacked vertically."""
    mids = {(s.start + s.end) // 2: s.index for s in result.shots[:max_rows]}
    frames: dict[int, np.ndarray] = {}
    with VideoReader(video, width=width) as vr:
        for i, f in enumerate(vr.frames()):
            if i in mids:
                frames[mids[i]] = f.rgb
            if len(frames) == len(mids):
                break
    rows = []
    for s in result.shots[:max_rows]:
        if s.index not in frames:
            continue
        img = frames[s.index]
        h = img.shape[0]
        pal = result.palettes[s.index] if s.index < len(result.palettes) else None
        strip = (palette_strip(Palette(np.array(pal["rgb"], np.uint8), np.array(pal["weights"]), None),
                               width=width * 2, height=h) if pal else np.zeros((h, width * 2, 3), np.uint8))
        label = Image.new("RGB", (220, h), (18, 18, 18))
        d = ImageDraw.Draw(label)
        cam = result.camera[s.index] if s.index < len(result.camera) else {}
        d.text((8, 6), f"shot {s.index}  {s.t0:7.2f}s  ({s.t1 - s.t0:4.1f}s)", fill=(230, 230, 230))
        if cam:
            d.text((8, 24), f"{cam['label']} {cam['direction']}".strip(), fill=(140, 220, 160))
            d.text((8, 42), f"speed {cam['speed']:.1f}  zoom {cam['zoom_rate']:+.2f}", fill=(150, 150, 150))
        rows.append(np.concatenate([img, np.asarray(label), strip], 1))
    if rows:
        Image.fromarray(np.concatenate(rows, 0)).save(out / "shots.jpg", quality=85)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-camera", action="store_true", help="skip camera-move classification")
    ap.add_argument("--k", type=int, default=5, help="colours per shot palette")
    args = ap.parse_args(argv)

    out = Path(args.out) / Path(args.video).stem
    out.mkdir(parents=True, exist_ok=True)
    result = analyse(args.video, with_camera=not args.no_camera, k=args.k)
    result.save(out / "shots.json")
    if result.barcode is not None:
        Image.fromarray(np.repeat(result.barcode, 160, axis=0)).save(out / "barcode.png")
    shot_sheet(args.video, result, out)

    kinds: dict[str, int] = {}
    for t in result.transitions:
        kinds[t.kind] = kinds.get(t.kind, 0) + 1
    lengths = [s.t1 - s.t0 for s in result.shots]
    print(f"{result.n_frames} frames, {len(result.shots)} shots "
          f"(median {np.median(lengths):.1f}s), transitions {kinds}")
    if result.camera:
        moves: dict[str, int] = {}
        for c in result.camera:
            moves[c["label"]] = moves.get(c["label"], 0) + 1
        print(f"camera moves: {moves}")
    print(f"timing: {result.seconds}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
