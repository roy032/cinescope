"""Find faces through a clip and draw them (the Phase 2 feature).

    python p2_detection/detect_clip.py trailers/Ted_Lasso_120s.mkv --ckpt p2_detection/checkpoints/centernet_r18.pt

Samples the clip at --fps, detects faces in every sampled frame, and writes
outputs/<clip>/faces.json (boxes per frame) and faces.jpg (a contact sheet with
the boxes drawn). Phase 3 turns these per-frame boxes into tracks and characters.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.detect.faces import FaceDetector  # noqa: E402
from cinescope.io.video import VideoReader  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=960, help="frames are decoded at this width")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--sheet", type=int, default=24, help="frames in the contact sheet")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    det = FaceDetector(args.ckpt, threshold=args.threshold)
    out = Path(args.out) / Path(args.video).stem
    out.mkdir(parents=True, exist_ok=True)
    frames, records, t0 = [], [], time.perf_counter()
    with VideoReader(args.video, width=args.width) as vr:
        for f in vr.frames(sample_fps=args.fps):
            r = det.detect([f.rgb])[0]
            records.append({"index": f.index, "t": round(f.t, 3),
                            "boxes": np.round(r["boxes"], 1).tolist(), "scores": np.round(r["scores"], 3).tolist()})
            frames.append((f.rgb, r["boxes"]))
    secs = time.perf_counter() - t0
    (out / "faces.json").write_text(json.dumps({"video": args.video, "fps": args.fps, "frames": records}))

    pick = np.linspace(0, len(frames) - 1, min(args.sheet, len(frames))).astype(int) if frames else []
    tiles = []
    for i in pick:
        rgb, boxes = frames[i]
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        for b in boxes:
            d.rectangle(list(map(float, b)), outline=(255, 214, 10), width=max(2, rgb.shape[1] // 300))
        tiles.append(im.resize((320, round(320 * rgb.shape[0] / rgb.shape[1]))))
    if tiles:
        cols = 4
        th = tiles[0].height
        sheet = Image.new("RGB", (cols * 320, -(-len(tiles) // cols) * th))
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % cols) * 320, (k // cols) * th))
        sheet.save(out / "faces.jpg", quality=85)
    n = sum(len(r["boxes"]) for r in records)
    print(f"{len(records)} frames, {n} faces, {secs / max(len(records), 1) * 1000:.0f} ms/frame -> {out}")
    return {"frames": len(records), "faces": n}


if __name__ == "__main__":
    main()
