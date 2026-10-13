"""Turn any videos you have (TV episodes, films, clips) into a shot-scale dataset:
find them, copy them into the project, cut them into shots with the Phase 0
detector, save three key frames per shot, and split by video.

    python p1_cnn/build_own_shots.py --search path/to/videos   # find, copy, cut, zip
    python p1_cnn/build_own_shots.py --videos data/episodes    # videos already in place

Files named like a series episode ("Show.Name.S02E07...", "Show Name 2x07") are
grouped by show; any other file is a group of its own.

Output (all git-ignored; the footage is never committed):
  data/episodes/<file>                        copies of the videos
  data/own_shots/frames/<episode>/<shot>_<k>.jpg   key frames at 25/50/75% of each shot
  data/own_shots/episodes/<episode>.json      shots of one video (resumable: done videos are skipped)
  data/own_shots/shots.csv                    split, episode, shot, start, end, frames
  kaggle_upload/dataset-own/own_shots.zip     the frames + shots.csv, for the Kaggle job

Episodes wider than 1920 px (4K, usually HDR) are skipped: decoding them on a
CPU is several times slower, and HDR decoded without tone mapping looks washed
out next to the SDR episodes.

Splits are by video, never by shot: neighbouring shots of one scene look
alike, and a shot-level split would leak the test set into training. Per show,
the last episode is test and the one before it validation; a show with fewer
than three episodes, or a single film or clip, is split by time instead
(first 70% / next 10% / last 20%).

Decoding is one pass per episode: frames are kept at 160 px for the cut
detector and every 6th frame is JPEG-encoded in memory at 448 px, so the key
frames need no second pass. Episodes run in parallel processes.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIDEO_EXT = {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".webm"}
SKIP_DIRS = {"$recycle.bin", "system volume information", "windows", "program files",
             "program files (x86)", "programdata", "appdata", "node_modules", ".git"}
EPISODE = re.compile(r"[sS](\d{1,2})\s*[eE](\d{1,3})|(?<!\d)(\d{1,2})x(\d{2})(?!\d)")
OUT = ROOT / "data" / "own_shots"
KEEP_EVERY = 6            # frames kept for key-frame selection (4 per second at 24 fps)
MIN_SHOT_FRAMES = 12      # half a second: shorter "shots" are mostly flashes and detector noise
KEY_WIDTH = 448
DARK = 12.0               # mean luma below this = black frame (fades, credits background)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def show_of(path: Path) -> str | None:
    """The series a file belongs to, from the name before its episode number
    ("Show.Name.S02E07.mkv" -> "show_name"); None for a file that is not an episode."""
    m = EPISODE.search(path.stem)
    if not m:
        return None
    return _slug(path.stem[:m.start()]) or None


def episode_id(path: Path) -> str:
    m = EPISODE.search(path.stem)
    show = show_of(path)
    if m and show:
        season, ep = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        return f"{show}_s{int(season):02d}e{int(ep):02d}"
    return _slug(path.stem)[:60] or "video"


def find_videos(search: Path, min_mb: float = 0) -> list[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(search):
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS and not d.startswith("$")]
        here = Path(dirpath).resolve()
        if any(here.is_relative_to(ROOT / d) for d in ("data", "outputs", ".venv", "kaggle_upload")):
            continue                  # our own copies and outputs, not new videos
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in VIDEO_EXT and p.stat().st_size >= min_mb * 1e6:
                found.append(p)
    return sorted(found)


def too_large(videos: list[Path], max_width: int) -> list[Path]:
    """Episodes wider than max_width (4K, usually HDR): slow to decode on a CPU, and HDR frames
    decoded without tone mapping look washed out next to the SDR episodes."""
    from cinescope.io.video import VideoReader

    out = []
    for v in videos:
        with VideoReader(v) as vr:
            if vr.info.width > max_width:
                out.append(v)
    return out


def copy_in(videos: list[Path], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    need = sum(v.stat().st_size for v in videos if not (dest / v.name).exists())
    free = shutil.disk_usage(dest).free
    if need > free - 5e9:
        print(f"not enough space on {dest.drive or dest} to copy {need / 1e9:.1f} GB "
              f"({free / 1e9:.1f} GB free): reading the videos where they are instead")
        return videos
    out = []
    for v in videos:
        d = dest / v.name
        if not d.exists() or d.stat().st_size != v.stat().st_size:
            print(f"copy {v} -> {d} ({v.stat().st_size / 1e9:.2f} GB)", flush=True)
            shutil.copy2(v, d)
        out.append(d)
    return out


def process_episode(video: str, ep: str) -> dict:
    """Cut one episode into shots and write its key frames. Runs in a worker process."""
    from PIL import Image

    from cinescope.io.video import Frame, VideoReader
    from cinescope.shots import cuts, features

    t0 = time.time()
    kept: dict[int, bytes] = {}
    luma: dict[int, float] = {}

    def frames_for_detector(vr):
        for f in vr.frames():
            if f.index % KEEP_EVERY == 0:
                buf = io.BytesIO()
                Image.fromarray(f.rgb).save(buf, "JPEG", quality=88)
                kept[f.index] = buf.getvalue()
                luma[f.index] = float(f.rgb.mean())
            h = max(1, round(f.rgb.shape[0] * 160 / f.rgb.shape[1]))
            small = np.asarray(Image.fromarray(f.rgb).resize((160, h), Image.BILINEAR))
            yield Frame(f.index, f.t, small)

    with VideoReader(video, width=KEY_WIDTH) as vr:
        fps = vr.info.fps
        feat = features.extract(frames_for_detector(vr), keep_thumbs=False)
    transitions = cuts.detect_transitions(feat)
    shots = cuts.shots_from_transitions(transitions, len(feat), feat.times)

    frame_dir = OUT / "frames" / ep
    frame_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in shots:
        if s.n_frames < MIN_SHOT_FRAMES:
            continue
        cands = [i for i in range(s.start + 2, s.end - 1) if i in kept and luma[i] >= DARK]
        if not cands:
            continue
        picks = sorted({cands[min(len(cands) - 1, int(p * len(cands)))] for p in (0.25, 0.5, 0.75)})
        rel = []
        for k, i in enumerate(picks):
            name = f"frames/{ep}/{s.index:04d}_{k}.jpg"
            (OUT / name).write_bytes(kept[i])
            rel.append(name)
        rows.append({"shot": f"{s.index:04d}", "start": s.start, "end": s.end,
                     "t0": round(s.t0, 3), "t1": round(s.t1, 3), "frames": rel})
    info = {"episode": ep, "show": show_of(Path(video)) or ep, "video": str(video), "fps": fps, "n_frames": len(feat),
            "n_shots_detected": len(shots), "shots": rows, "seconds": round(time.time() - t0, 1)}
    (OUT / "episodes").mkdir(parents=True, exist_ok=True)
    (OUT / "episodes" / f"{ep}.json").write_text(json.dumps(info), encoding="utf-8")
    return info


def assign_splits(episodes: list[dict]) -> dict[tuple[str, str], str]:
    """(episode, shot) -> split. By episode per show; by time within a show with < 3 episodes."""
    split: dict[tuple[str, str], str] = {}
    by_show: dict[str, list[dict]] = {}
    for e in episodes:
        by_show.setdefault(e.get("show", e["episode"]), []).append(e)
    for eps in by_show.values():
        eps.sort(key=lambda e: e["episode"])
        if len(eps) >= 3:
            roles = ["train"] * (len(eps) - 2) + ["val", "test"]
            for e, r in zip(eps, roles, strict=True):
                for s in e["shots"]:
                    split[(e["episode"], s["shot"])] = r
        else:
            for e in eps:
                dur = max(s["t1"] for s in e["shots"]) if e["shots"] else 1.0
                for s in e["shots"]:
                    f = s["t0"] / dur
                    split[(e["episode"], s["shot"])] = "train" if f < 0.7 else "val" if f < 0.8 else "test"
    return split


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", help="folder to search (recursively) for video files")
    ap.add_argument("--min-mb", type=float, default=0, help="with --search: ignore videos smaller than this")
    ap.add_argument("--videos", default=str(ROOT / "data" / "episodes"), help="where the video copies live")
    ap.add_argument("--no-copy", action="store_true", help="read the videos where they were found")
    ap.add_argument("--pack-only", action="store_true",
                    help="skip finding and cutting: rebuild shots.csv and the zip from every cut episode")
    ap.add_argument("--max-width", type=int, default=1920, help="skip videos wider than this (4K/HDR)")
    ap.add_argument("--workers", type=int, default=max(1, min(3, (os.cpu_count() or 2) // 2)))
    args = ap.parse_args(argv)

    if args.pack_only:
        ids = sorted(p.stem for p in (OUT / "episodes").glob("*.json"))
        if not ids:
            sys.exit(f"no cut videos in {OUT / 'episodes'}")
        pack(ids)
        return
    if args.search:
        found = find_videos(Path(args.search), args.min_mb)
        if not found:
            sys.exit(f"no video files found under {args.search}")
        print(f"found {len(found)} videos under {args.search}")
        big = too_large(found, args.max_width)
        for v in big:
            print(f"  skipped (wider than {args.max_width} px): {v.name}")
        found = [v for v in found if v not in big]
        videos = found if args.no_copy else copy_in(found, Path(args.videos))
    else:
        videos = sorted(p for p in Path(args.videos).glob("*") if p.suffix.lower() in VIDEO_EXT)
        videos = [v for v in videos if v not in too_large(videos, args.max_width)]
    if not videos:
        sys.exit("no videos to process")

    ids = [episode_id(v) for v in videos]
    if len(set(ids)) != len(ids):
        ids = [f"{i}_{n}" for n, i in enumerate(ids)]
    todo = [(str(v), e) for v, e in zip(videos, ids, strict=True) if not (OUT / "episodes" / f"{e}.json").exists()]
    print(f"{len(videos)} videos, {len(videos) - len(todo)} already cut, {len(todo)} to do "
          f"({args.workers} in parallel; roughly 5-15 minutes each)", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(process_episode, v, e): e for v, e in todo}
        for f in as_completed(futs):
            info = f.result()
            print(f"  {info['episode']}: {len(info['shots'])} shots kept of {info['n_shots_detected']} "
                  f"({info['seconds'] / 60:.1f} min)", flush=True)

    pack(ids)


def pack(ids: list[str]) -> None:
    """shots.csv (with the episode-level splits) and the zip for Kaggle, from the cut episodes."""
    episodes = [json.loads((OUT / "episodes" / f"{e}.json").read_text(encoding="utf-8")) for e in ids]
    for e in episodes:
        e.setdefault("show", show_of(Path(e["video"])) or e["episode"])
    split = assign_splits(episodes)
    with (OUT / "shots.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "episode", "shot", "start", "end", "frames"])
        for e in episodes:
            for s in e["shots"]:
                w.writerow([split[(e["episode"], s["shot"])], e["episode"], s["shot"], s["start"], s["end"],
                            ";".join(s["frames"])])
    counts: dict[str, int] = {}
    for v in split.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"shots.csv: {sum(counts.values())} shots {counts}")

    zdir = ROOT / "kaggle_upload" / "dataset-own"
    zdir.mkdir(parents=True, exist_ok=True)
    zpath = zdir / "own_shots.zip"
    tmp = zdir / "own_shots.zip.part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        z.write(OUT / "shots.csv", "shots.csv")
        for e in episodes:
            for s in e["shots"]:
                for rel in s["frames"]:
                    z.write(OUT / rel, rel)
    tmp.replace(zpath)
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
