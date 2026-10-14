"""Per-shot breakdown of any clip: scene type, people and poses, action events.

    python p6_pose_video/breakdown.py trailers/goldeneye.mp4
    python p6_pose_video/breakdown.py clip.mp4 --pose yolo --scene clip           # library models only
    # -> outputs/<clip>/breakdown.json, outputs/<clip>/breakdown.jpg

Pipeline:
  1. shots           Phase 0 cut detection (cached in outputs/shots/)
  2. poses           frames at --pose-fps inside each shot;
                     ours: YOLOv8n person boxes -> SimpleBaseline (top-down, from scratch)
                     yolo: YOLOv8n-pose (library)
  3. tracks, events  IoU tracking within the shot; strikes / falls / running
                     (cinescope/pose/events.py)
  4. scene type      ours: VideoMAE-pretrained video ViT, fine-tuned on CLIP teacher labels
                     clip: OpenCLIP ViT-B/32 zero-shot (library)
  5. visual          one tile per shot: the middle pose frame with skeletons,
                     the scene type and the shot's events

Models that are missing fall back to the library ones, and the JSON records
which ran.
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

from cinescope.pose import SKELETON  # noqa: E402
from cinescope.pose.events import EventParams, detect_events  # noqa: E402
from cinescope.pose.track import track_people  # noqa: E402
from cinescope.video.clips import clip_name, resolve, shots, windows  # noqa: E402
from cinescope.video.frames import frames_at  # noqa: E402
from cinescope.video.scenes import CLASSES, softmax  # noqa: E402

COLORS = [(255, 80, 80), (80, 200, 255), (255, 210, 60), (120, 255, 120), (255, 120, 255), (255, 160, 60)]


class PoseBackend:
    def __init__(self, kind: str, ckpt: str | None):
        from cinescope.pose.yolo import YoloPose

        self.kind = kind
        if kind == "ours":
            from cinescope.pose.simplebaseline import load_checkpoint

            self.det = YoloPose(pose=False, conf=0.4)
            self.net, ck = load_checkpoint(ckpt)
            self.input_hw = tuple(ck.get("input_hw", (128, 96)))
        else:
            self.det = YoloPose(pose=True, conf=0.4)

    def __call__(self, frames: list[np.ndarray]):
        people = self.det(frames)
        if self.kind == "ours":
            from cinescope.pose.simplebaseline import predict

            for f, p in zip(frames, people, strict=True):
                if len(p):
                    p.kpts = predict(self.net, f, p.boxes, self.input_hw)
        return people


class SceneBackend:
    def __init__(self, kind: str, ckpt: str | None):
        self.kind = kind
        if kind == "ours":
            import torch

            from p6_pose_video.train_scenes import make_model

            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.model = make_model(ck["kind"], None)
            self.model.load_state_dict(ck["model"])
            self.model_kind = ck["kind"]
        else:
            from cinescope.video.scenes import ClipTeacher

            self.teacher = ClipTeacher()

    def window_probs(self, path: Path, wins: list[tuple[int, float, float]]) -> np.ndarray:
        from p6_pose_video.build_scene_data import small, window_times

        times, spans = [], []
        for _, a, b in wins:
            ct, tt = window_times(a, b)
            spans.append((len(times), len(ct), len(tt)))
            times += ct + tt
        frames = frames_at(path, times, width=320)
        if self.kind == "ours":
            from p6_pose_video.train_scenes import predict_windows

            stack = np.stack([np.stack([small(f) for f in frames[i:i + nc]]) for i, nc, _ in spans])
            return predict_windows(self.model, self.model_kind, stack)
        return np.stack([softmax(self.teacher.frame_logits(frames[i + nc:i + nc + nt]).mean(0))
                         for i, nc, nt in spans])


def draw_people(img: Image.Image, boxes, kpts, ids=None, scale: float = 1.0, min_conf: float = 0.3) -> None:
    dr = ImageDraw.Draw(img)
    for j, (b, k) in enumerate(zip(boxes, kpts, strict=True)):
        col = COLORS[(ids[j] if ids is not None else j) % len(COLORS)]
        b = np.asarray(b) * scale
        dr.rectangle(list(b), outline=col, width=1)
        k = np.asarray(k)
        for a, c in SKELETON:
            if k[a, 2] >= min_conf and k[c, 2] >= min_conf:
                dr.line([tuple(k[a, :2] * scale), tuple(k[c, :2] * scale)], fill=col, width=2)
        for x, y, c in k:
            if c >= min_conf:
                dr.ellipse([x * scale - 2, y * scale - 2, x * scale + 2, y * scale + 2], fill=col)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--pose", choices=["ours", "yolo"], default="ours")
    ap.add_argument("--pose-ckpt", default="outputs/p6/pose_sb.pt")
    ap.add_argument("--scene", choices=["ours", "clip"], default="ours")
    ap.add_argument("--scene-ckpt", default="outputs/p6/scene_mae_ft.pt")
    ap.add_argument("--pose-fps", type=float, default=8.0)
    ap.add_argument("--max-width", type=int, default=960)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    path = resolve(args.clip)
    name = clip_name(path)
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    info = shots(path)
    sh = info["shots"]
    if args.pose == "ours" and not Path(args.pose_ckpt).exists():
        print(f"no pose checkpoint at {args.pose_ckpt}: using YOLOv8n-pose")
        args.pose = "yolo"
    if args.scene == "ours" and not Path(args.scene_ckpt).exists():
        print(f"no scene checkpoint at {args.scene_ckpt}: using OpenCLIP zero-shot")
        args.scene = "clip"

    # --- poses at pose-fps inside every shot
    from cinescope.io.video import VideoReader

    with VideoReader(path) as vr:
        W = vr.info.width
    width = min(W, args.max_width)
    sample_t, sample_shot = [], []
    for i, s in enumerate(sh):
        ts = np.arange(s["t0"] + 0.5 / args.pose_fps, s["t1"], 1 / args.pose_fps)
        if not len(ts):
            ts = np.array([0.5 * (s["t0"] + s["t1"])])
        sample_t += list(ts)
        sample_shot += [i] * len(ts)
    frames = frames_at(path, sample_t, width=width if width != W else None)
    pose = PoseBackend(args.pose, args.pose_ckpt)
    people = []
    for i in range(0, len(frames), 8):
        people += pose(frames[i:i + 8])
    t_pose = time.time() - t_start

    # --- scene types per window, averaged per shot
    scene = SceneBackend(args.scene, args.scene_ckpt)
    wins = windows(sh, 2.0, min_seconds=0.0)
    probs = scene.window_probs(path, wins)
    t_scene = time.time() - t_start - t_pose

    params = EventParams()
    sample_shot = np.array(sample_shot)
    result = {"video": str(path), "fps": info["fps"], "pose_model": args.pose, "scene_model": args.scene,
              "pose_fps": args.pose_fps, "event_params": params.__dict__, "shots": []}
    tiles = []
    for i, s in enumerate(sh):
        idx = np.where(sample_shot == i)[0]
        ts = [sample_t[j] for j in idx]
        tracks = track_people(ts, [people[j].boxes for j in idx], [people[j].kpts for j in idx])
        events = detect_events(tracks, params)
        p = probs[[k for k, w in enumerate(wins) if w[0] == i]].mean(0)
        counts = [len(people[j]) for j in idx]
        heights = [float((b[3] - b[1]) / frames[j].shape[0]) for j in idx for b in people[j].boxes]
        vis = [float((k[:, 2] >= 0.3).sum()) for j in idx for k in people[j].kpts]
        result["shots"].append({
            "index": i, "t0": round(s["t0"], 3), "t1": round(s["t1"], 3),
            "scene_type": CLASSES[int(p.argmax())],
            "scene_probs": {c: round(float(v), 3) for c, v in zip(CLASSES, p, strict=True)},
            "people": {"max_in_frame": int(max(counts)) if counts else 0,
                       "mean_in_frame": round(float(np.mean(counts)), 2) if counts else 0.0,
                       "tracks": len(tracks),
                       "median_height_frac": round(float(np.median(heights)), 3) if heights else None,
                       "mean_visible_keypoints": round(float(np.mean(vis)), 1) if vis else None},
            "events": [e.to_dict() for e in events],
        })
        # tile: the middle sample of the shot, with its skeletons
        j = idx[len(idx) // 2]
        im = Image.fromarray(frames[j])
        sc = 320 / im.width
        im = im.resize((320, round(im.height * sc)))
        tid = {}
        for tr in tracks:
            if sample_t[j] in tr.times:
                k = tr.times.index(sample_t[j])
                tid[tuple(np.round(tr.boxes[k], 2))] = tr.id
        ids = [tid.get(tuple(np.round(b, 2)), q) for q, b in enumerate(people[j].boxes)]
        draw_people(im, people[j].boxes, people[j].kpts, ids, sc)
        tiles.append((im, s, result["shots"][-1]))
    t_total = time.time() - t_start
    result["timing_s"] = {"poses": round(t_pose, 1), "scene_types": round(t_scene, 1), "total": round(t_total, 1)}
    (out_dir / "breakdown.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    _sheet(tiles, out_dir / "breakdown.jpg")
    n_ev = sum(len(s["events"]) for s in result["shots"])
    print(f"{name}: {len(sh)} shots, {len(sample_t)} pose frames, {n_ev} events, {t_total:.0f} s "
          f"-> {out_dir / 'breakdown.json'}, breakdown.jpg")
    return result


def _sheet(tiles, path: Path, cols: int = 4) -> None:
    w = 320
    hs = [t[0].height for t in tiles]
    h = max(hs) + 30
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * (w + 4) + 4, rows * (h + 4) + 4), (18, 18, 18))
    dr = ImageDraw.Draw(sheet)
    for n, (im, s, rec) in enumerate(tiles):
        x, y = 4 + (n % cols) * (w + 4), 4 + (n // cols) * (h + 4)
        sheet.paste(im, (x, y))
        ev = ", ".join(f"{e['type']}@{e['t']:.1f}" for e in rec["events"][:3]) or "no events"
        dr.text((x + 2, y + im.height + 2), f"#{rec['index']} {s['t0']:.1f}-{s['t1']:.1f}s  {rec['scene_type']}",
                fill=(255, 230, 120))
        dr.text((x + 2, y + im.height + 15), f"people {rec['people']['max_in_frame']}  {ev}", fill=(220, 220, 220))
    sheet.save(path, quality=85)


if __name__ == "__main__":
    main()
