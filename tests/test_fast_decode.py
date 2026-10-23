"""Decoding speed-ups: swscale resizes that match the PIL ones the detectors were
built on, lazy RGB conversion, and checkpoints paced by wall-clock time."""
import numpy as np
import pytest

pytest.importorskip("av")

from cinescope.io.video import Picture, VideoReader, _resize, write_video  # noqa: E402
from cinescope.pipeline import Interrupted, Options, run  # noqa: E402


def smooth_clip(path, n=24, w=640, h=360):
    """Gradients and soft blobs: the kind of content where resamplers should agree."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frames = []
    for k in range(n):
        r = 127 + 100 * np.sin(xx / 60 + k / 4)
        g = 127 + 100 * np.cos(yy / 45 - k / 5)
        b = 255 * np.exp(-((xx - 10 * k - 200) ** 2 + (yy - 180) ** 2) / 4000)
        frames.append(np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8))
    write_video(path, np.stack(frames), 24.0)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("dec") / "smooth.mp4"
    smooth_clip(p)
    return p


@pytest.mark.parametrize("width", [96, 160, 320, 480, 640, 960])
def test_swscale_matches_pil_shape_and_pixels(clip, width):
    with VideoReader(clip) as vr:
        _, _, pic = next(iter(vr.pictures()))
        full = pic.rgb()
        fast = pic.rgb(width)
    ref = _resize(full, width)
    assert fast.shape == ref.shape and fast.dtype == np.uint8
    assert np.abs(fast.astype(float) - ref).mean() < 3.0     # grey levels, out of 255


def test_reader_width_uses_the_same_resize(clip):
    with VideoReader(clip, width=160) as vr:
        a = [f.rgb for f in vr.frames()]
    with VideoReader(clip) as vr:
        b = [pic.rgb(160) for _, _, pic in vr.pictures()]
    assert len(a) == len(b) == 24
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_picture_converts_each_width_once():
    calls = []

    class Fake:
        width, height = 64, 36

        def to_ndarray(self, **kw):
            calls.append(kw.get("width"))
            return np.zeros((kw.get("height", 36), kw.get("width", 64), 3), np.uint8)

    pic = Picture(Fake(), 36, 64, True)
    for _ in range(3):
        pic.rgb()
        pic.rgb(16)
        pic.rgb(64)                           # the source width is the full frame
    assert calls == [None, 32]               # full once; 16 px via a 32 px swscale step, once


def test_sampled_frames_skip_conversion(clip, monkeypatch):
    n = []
    orig = Picture.rgb
    monkeypatch.setattr(Picture, "rgb", lambda self, w=None: n.append(1) or orig(self, w))
    with VideoReader(clip) as vr:
        got = list(vr.frames(sample_fps=4))
    assert len(got) == 4 and len(n) == 4     # 24 frames decoded, only the 4 kept became RGB


def test_checkpoints_are_paced_by_wall_time(tmp_path):
    from tests.test_week25 import synthetic_clip

    video = tmp_path / "synth.mp4"
    synthetic_clip(video)
    for name, wall, expect in (("slow", 3600.0, False), ("fast", 0.0, True)):
        opts = Options(checkpoint_s=0.2, checkpoint_wall_s=wall)
        with pytest.raises(Interrupted):
            run(video, ["shots"], tmp_path / name, options=opts, log=None, stop=lambda n, t: n == 60)
        assert (tmp_path / name / "synth" / "stages" / "_decode" / "checkpoint.pkl").exists() is expect
