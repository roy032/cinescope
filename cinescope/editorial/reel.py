"""Render a highlights reel: the report's highlight moments cut together as one MP4.

    render_reel("clip.mp4", report["extra"]["highlights"]["moments"], "clip_highlights.mp4")

Each moment is decoded from the source (cinescope.io.video.VideoReader, which
repairs B-frame timestamps and seeks exactly) and re-encoded: H.264 when the
FFmpeg build has it, MPEG-4 Part 2 otherwise. The sound of each moment is cut
to the length of its frames, so picture and sound stay in sync across every
join, and faded in and out over a few milliseconds so the joins do not click.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from ..io.video import VideoReader

AUDIO_RATE = 48000
FADE_S = 0.012


def _video_codec(container, fps: Fraction):
    import av

    for name in ("libx264", "h264", "mpeg4"):
        try:
            s = container.add_stream(name, rate=fps)
        except (av.FFmpegError, ValueError, LookupError):
            continue
        if name != "mpeg4":
            s.options = {"crf": "21", "preset": "veryfast"}
        else:
            s.bit_rate = 6_000_000
        return s
    raise RuntimeError("no H.264 or MPEG-4 encoder in this FFmpeg build")


def _audio(path: str, t0: float, t1: float, rate: int = AUDIO_RATE) -> np.ndarray | None:
    """(2, n) float32 samples of [t0, t1), or None when the file has no sound."""
    import av

    with av.open(path) as c:
        if not c.streams.audio:
            return None
        st = c.streams.audio[0]
        res = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
        if t0 > 0:
            c.seek(int(max(t0 - 0.5, 0) / st.time_base), stream=st, backward=True)
        chunks, start = [], None
        for packet in c.demux(st):
            try:
                frames = packet.decode()
            except av.FFmpegError:
                continue
            for fr in frames:
                if fr.pts is None:
                    continue
                ft = float(fr.pts * st.time_base)
                for r in res.resample(fr):
                    if start is None:
                        start = ft
                    chunks.append(r.to_ndarray().reshape(2, -1))
                if ft > t1 + 0.5:
                    break
            else:
                continue
            break
    if not chunks or start is None:
        return None
    x = np.concatenate(chunks, axis=1)
    a = int(round((t0 - start) * rate))
    n = int(round((t1 - t0) * rate))
    if a < 0:                                    # the stream starts after t0: pad with silence
        x = np.concatenate([np.zeros((2, -a), np.float32), x], axis=1)
        a = 0
    seg = x[:, a:a + n]
    if seg.shape[1] < n:
        seg = np.concatenate([seg, np.zeros((2, n - seg.shape[1]), np.float32)], axis=1)
    return seg.astype(np.float32)


def _fade(x: np.ndarray, rate: int = AUDIO_RATE) -> np.ndarray:
    k = min(int(FADE_S * rate), x.shape[1] // 2)
    if k > 0:
        ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
        x = x.copy()
        x[:, :k] *= ramp
        x[:, -k:] *= ramp[::-1]
    return x


def render_reel(video: str | Path, moments: list[dict], dest: str | Path, width: int | None = None,
                progress=None) -> Path:
    """Cut `moments` ([{"t0", "t1"}], in order) out of `video` into one MP4 at `dest`.
    width: scale the picture down to this width (None: the source size)."""
    import av

    if not moments:
        raise ValueError("no highlight moments to render")
    video, dest = str(video), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with VideoReader(video) as vr:
        info = vr.info
    fps = Fraction(info.fps).limit_denominator(1001)
    w = width if width and width < info.width else info.width
    h = int(round(info.height * w / info.width))
    w, h = w - w % 2, h - h % 2                  # yuv420p needs even sizes
    tmp = dest.with_suffix(".part.mp4")
    total = sum(m["t1"] - m["t0"] for m in moments)
    done = 0.0
    with av.open(str(tmp), "w") as out:
        vs = _video_codec(out, fps)
        vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
        has_audio = _audio(video, moments[0]["t0"], min(moments[0]["t0"] + 0.1, moments[0]["t1"])) is not None
        aus = out.add_stream("aac", rate=AUDIO_RATE) if has_audio else None
        if aus is not None:
            aus.layout = "stereo"
        n_frames, n_samples = 0, 0
        for m in moments:
            first_t, k = None, 0
            with VideoReader(video) as vr:
                for _, ft, pic in vr.pictures(start_s=float(m["t0"]), end_s=float(m["t1"])):
                    if first_t is None:
                        first_t = ft
                    rgb = pic.rgb(w)[:h, :w]           # swscale does the scaling (fast)
                    if rgb.shape[:2] != (h, w):      # rounding: pad a row / column
                        rgb = np.pad(rgb, ((0, h - rgb.shape[0]), (0, w - rgb.shape[1]), (0, 0)), mode="edge")
                    vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
                    vf.pts = n_frames + k
                    vf.time_base = 1 / fps
                    for p in vs.encode(vf):
                        out.mux(p)
                    k += 1
            if aus is not None and k:
                seg = _audio(video, first_t, first_t + k / float(fps))
                if seg is None:
                    seg = np.zeros((2, int(round(k / float(fps) * AUDIO_RATE))), np.float32)
                seg = _fade(seg)
                step = 1024
                for i in range(0, seg.shape[1], step):
                    chunk = np.ascontiguousarray(seg[:, i:i + step])
                    af = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
                    af.sample_rate = AUDIO_RATE
                    af.pts = n_samples + i
                    af.time_base = Fraction(1, AUDIO_RATE)
                    for p in aus.encode(af):
                        out.mux(p)
                n_samples += seg.shape[1]
            n_frames += k
            done += m["t1"] - m["t0"]
            if progress is not None:
                progress(done / max(total, 1e-6))
        for p in vs.encode():
            out.mux(p)
        if aus is not None:
            for p in aus.encode():
                out.mux(p)
    if n_frames == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError("the highlight moments contain no frames")
    tmp.replace(dest)
    return dest
