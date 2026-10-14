"""The Phase 3 feature: clip -> characters, when each is on screen, who talks.

    from cinescope.faces.characters import character_timeline
    result = character_timeline("clip.mp4", "outputs/clip")

Steps (each cached in out_dir, so reruns are fast):
  faces per frame (YuNet or the Phase 2 CenterNet) -> shot cuts (Phase 0) ->
  tracks (ByteTrack-style, reset at cuts) -> aligned crops -> one embedding
  per track (ours or SFace) -> agglomerative clustering with cannot-links ->
  characters ordered by screen time -> per-frame speaking score (ours or
  Light-ASD) -> intervals.

Tracks shorter than `min_track_s` or with a low mean detector score are
dropped first: a quarter-second "face" is far more often a false positive
than a character.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .clip import (
    TrackCrops,
    detect_clip,
    extract_crops,
    load_crops,
    save_crops,
    shot_starts,
    track_clip,
)
from .cluster import agglomerative, auto_threshold, cooccurring_pairs, cosine_distances


def _intervals(t: np.ndarray, on: np.ndarray, gap: float, min_len: float = 0.0) -> list[list[float]]:
    """Boolean samples at times t -> merged [start, end] intervals."""
    out: list[list[float]] = []
    for ti, o in zip(t, on, strict=True):
        if not o:
            continue
        if out and ti - out[-1][1] <= gap:
            out[-1][1] = float(ti)
        else:
            out.append([float(ti), float(ti)])
    return [[round(a, 3), round(b + gap / 2, 3)] for a, b in out if b - a >= min_len]


def _embedder(kind: str, ckpt: str | Path | None):
    from .embed import SFaceEmbedder, embed, load

    if kind == "sface":
        return SFaceEmbedder()
    m = load(str(ckpt))
    return lambda x: embed(m, x)


def character_timeline(video: str | Path, out_dir: str | Path, detector: str = "yunet",
                       det_ckpt: str | None = None, embedder: str = "ours", embed_ckpt: str | None = None,
                       asd: str = "ours", min_track_s: float = 0.4, min_score: float = 0.6,
                       speak_threshold: float | None = None, log=print) -> dict:
    from .detectors import make_detector

    video, out = str(video), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / f"faces_{detector}.npz"
    dets = detect_clip(video, None if cache.exists() else make_detector(detector, ckpt=det_ckpt), cache=cache,
                       log=log)
    sp = out / "shots.json"
    if sp.exists():
        cuts = json.loads(sp.read_text())["starts"]
    else:
        cuts = shot_starts(video)
        sp.write_text(json.dumps({"video": video, "starts": cuts}))
    tracks = track_clip(dets, cuts)
    fps, times = dets.fps, dets.times
    tracks = [t for t in tracks if len(t) >= min_track_s * fps and np.mean(t.scores) >= min_score]
    cc = out / f"crops_{detector}_timeline.npz"
    if cc.exists():
        crops = load_crops(cc)
        keep = {t.id for t in tracks}
        crops = [c for c in crops if c.track.id in keep]
    else:
        crops = extract_crops(video, dets, tracks, max_rgb=24)
        save_crops(cc, crops)
    log(f"  {len(dets.frames)} frames, {len(cuts)} shots, {len(crops)} tracks kept")

    # ---- identities
    emb = _embedder(embedder, embed_ckpt)
    E, same = [], []
    for c in crops:
        e = emb(c.rgb)
        m = e.mean(0)
        E.append(m / max(np.linalg.norm(m), 1e-12))
        if len(e) >= 4:
            a, b = e[: len(e) // 2].mean(0), e[len(e) // 2:].mean(0)
            same.append(1 - float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
    E = np.asarray(E).reshape(len(crops), -1)
    D = cosine_distances(E) if len(E) else np.zeros((0, 0))
    cl = cooccurring_pairs([c.track.frames for c in crops])
    thr = auto_threshold(np.asarray(same), np.asarray([D[a, b] for a, b in cl]))
    if not np.isfinite(thr):
        thr = 0.5 if embedder == "sface" else 0.3
    labels = agglomerative(D, thr, cannot_link=cl)

    # ---- who is talking
    speak = _speaking(video, crops, times, fps, asd)
    if speak_threshold is None:
        speak_threshold = 0.25 if asd == "ours" else 0.0

    chars = []
    for k in set(labels.tolist()):
        members = [crops[i] for i in np.flatnonzero(labels == k)]
        frames = np.unique(np.concatenate([m.track.frames for m in members]))
        talk = np.unique(np.concatenate([np.asarray(m.track.frames)[speak[m.track.id] >= speak_threshold]
                                         for m in members]))
        best = max(members, key=lambda m: np.median([b[3] - b[1] for b in m.track.boxes]))
        chars.append({
            "tracks": [m.track.id for m in members],
            "screen_time_s": round(len(frames) / fps, 2),
            "speaking_time_s": round(len(talk) / fps, 2),
            "on_screen": _intervals(times[frames], np.ones(len(frames), bool), 1.5 / fps),
            "speaking": _intervals(times[talk], np.ones(len(talk), bool), 3 / fps, min_len=0.2) if len(talk) else [],
            "_thumb": best.rgb[len(best.rgb) // 2],
        })
    chars.sort(key=lambda c: -c["screen_time_s"])
    for i, c in enumerate(chars):
        c["id"] = f"C{i + 1}"
    result = {
        "video": video, "fps": fps, "duration_s": round(float(times[-1] + 1 / fps), 2) if len(times) else 0,
        "detector": detector, "embedder": embedder, "asd": asd, "cluster_threshold": round(float(thr), 4),
        "n_shots": len(cuts), "n_tracks": len(crops),
        "characters": [{k: v for k, v in c.items() if not k.startswith("_")} for c in chars],
        "tracks": [{**c.track.to_dict(), "character": chars_id(chars, c.track.id),
                    "speaking_score": np.round(speak[c.track.id], 3).tolist()} for c in crops],
    }
    result["_thumbs"] = [c["_thumb"] for c in chars]
    return result


def chars_id(chars: list[dict], track_id: int) -> str:
    for c in chars:
        if track_id in c["tracks"]:
            return c.get("id", "")
    return ""


def _speaking(video: str, crops: list[TrackCrops], times: np.ndarray, fps: float, kind: str) -> dict:
    from ..speaker.audio import load_audio, vad

    audio = load_audio(video)
    if audio is None:
        return {c.track.id: np.zeros(len(c.track)) for c in crops}
    if kind == "lightasd":
        from ..speaker.lightasd import LightASD

        m = LightASD()
        return {c.track.id: m.score(c.asd, times[np.asarray(c.track.frames)], audio) for c in crops}
    from ..speaker.active import score_track

    v = vad(audio)
    return {c.track.id: score_track(c.gray, c.track.frames, times, v, fps)["score"] for c in crops}


def plot_timeline(result: dict, path: str | Path, max_rows: int = 20) -> None:
    """One row per character: on-screen spans, talking spans on top, a face thumbnail at the left."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chars = result["characters"][:max_rows]
    thumbs = result.get("_thumbs", [])[:max_rows]
    n = max(1, len(chars))
    dur = max(result["duration_s"], 1e-3)
    fig, ax = plt.subplots(figsize=(11, 0.55 * n + 1.3), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    on_c, talk_c = "#2a78d6", "#eb6834"         # categorical slots 1 and 2
    for i, c in enumerate(chars):
        y = n - 1 - i
        ax.broken_barh([(a, max(b - a, 0.05)) for a, b in c["on_screen"]], (y - 0.32, 0.64), color=on_c,
                       linewidth=0)
        if c["speaking"]:
            ax.broken_barh([(a, max(b - a, 0.05)) for a, b in c["speaking"]], (y - 0.14, 0.28), color=talk_c,
                           linewidth=0)
        ax.text(dur * 1.005, y, f"{c['screen_time_s']:.1f} s  ({c['speaking_time_s']:.1f} s talking)",
                va="center", fontsize=8, color="#52514e")
        if i < len(thumbs):
            from matplotlib.offsetbox import AnnotationBbox, OffsetImage

            ab = AnnotationBbox(OffsetImage(thumbs[i], zoom=0.24), (0, y), xybox=(-34, 0), xycoords="data",
                                boxcoords="offset points", frameon=False, annotation_clip=False)
            ax.add_artist(ab)
    ax.set_yticks(range(n))
    ax.set_yticklabels([c["id"] for c in chars][::-1], fontsize=9, color="#0b0b0b")
    ax.tick_params(axis="y", pad=58, length=0)
    ax.set_xlim(0, dur)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("time (s)", color="#52514e")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b0afa8")
    ax.grid(axis="x", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=on_c, label="on screen"), Patch(color=talk_c, label="talking")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    ax.set_title(f"{Path(result['video']).name}: {len(result['characters'])} characters "
                 f"({result['embedder']} embeddings, {result['asd']} speaker detection)",
                 fontsize=10, color="#0b0b0b", pad=22)
    fig.tight_layout()
    fig.subplots_adjust(right=0.84)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
