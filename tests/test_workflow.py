"""Real-world workflow: clips that share a file name keep separate results, and
files a browser cannot play get a browser-playable copy on the same timeline."""
import time
from pathlib import Path

import numpy as np
import pytest

from cinescope.pipeline.library import clip_names

pytest.importorskip("av")


def test_clip_names_keep_first_stem_and_disambiguate():
    files = [Path("a/ep01.mp4"), Path("b/ep01.mkv"), Path("c/ep02.mp4"), Path("b/ep01.mp4")]
    assert clip_names(files) == ["ep01", "ep01-b", "ep02", "ep01-b-2"]
    assert clip_names([Path("x/film.mp4")], taken={"film"}) == ["film-x"]


def test_batch_with_same_names_keeps_both(tmp_path):
    from cinescope.pipeline.__main__ import main
    from tests.test_week25 import synthetic_clip

    for season in ("Season 1", "Season 2"):
        (tmp_path / season).mkdir()
        synthetic_clip(tmp_path / season / "ep01.mp4")
    out = tmp_path / "out"
    res = main([str(tmp_path / "Season 1"), str(tmp_path / "Season 2"), "--stages", "shots",
                "--out", str(out), "--quiet", "--no-library"])
    assert [r["clip"] for r in res["reports"]] == ["ep01", "ep01-Season 2"]
    assert (out / "ep01" / "report.json").exists() and (out / "ep01-Season 2" / "report.json").exists()


def _frames(path):
    from cinescope.io.video import VideoReader

    with VideoReader(path) as vr:
        return [(round(f.t, 3), f.rgb.mean()) for f in vr.frames()]


def test_proxy_keeps_every_frame_in_order(tmp_path):
    from cinescope.editorial.reel import render_proxy
    from tests.test_week25 import synthetic_clip

    src = tmp_path / "clip.mp4"
    synthetic_clip(src)                              # 90 frames, three colours
    dest = render_proxy(src, tmp_path / "proxy.mp4", width=64, chunk_s=1.0)   # 4 back-to-back chunks
    a, b = _frames(src), _frames(dest)
    assert len(a) == len(b) == 90
    assert np.allclose([t for t, _ in a], [t for t, _ in b], atol=0.05)
    assert np.corrcoef([m for _, m in a], [m for _, m in b])[0, 1] > 0.99      # same picture, same order


def test_app_makes_and_serves_the_proxy(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from app.backend.main import create_app
    from tests.test_week25 import synthetic_clip

    clips = tmp_path / "clips"
    clips.mkdir()
    synthetic_clip(clips / "clip.avi")
    c = TestClient(create_app([clips], tmp_path / "out"))
    assert c.get("/api/clips/clip/proxy").json()["status"] == "none"
    original = c.get("/api/clips/clip/video").content
    assert c.post("/api/clips/clip/proxy").json()["status"] in ("running", "done")
    for _ in range(300):
        st = c.get("/api/clips/clip/proxy").json()
        if st["status"] != "running":
            break
        time.sleep(0.05)
    assert st["status"] == "done", st
    r = c.get("/api/clips/clip/video")
    assert r.headers["content-type"] == "video/mp4" and r.content != original
    assert c.get("/api/clips/clip/video?original=1").content == original
    assert c.post("/api/clips/clip/proxy").json()["status"] == "done"          # made once
    assert c.post("/api/clips/%2e%2e/proxy").status_code == 400


def test_app_lists_same_named_clips(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from app.backend.main import create_app

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "film.mp4").write_bytes(b"\0" * 2048)
    (b / "film.mp4").write_bytes(b"\0" * 4096)
    c = TestClient(create_app([a, b], tmp_path / "out"))
    rows = {r["clip"]: r for r in c.get("/api/clips").json()}
    assert set(rows) == {"film", "film-b"}
    assert len(c.get("/api/clips/film-b/video").content) == 4096
