"""The web app's guard rails: clip names cannot leave the outputs folder, uploads
have a size cap and never leave half files, exports render once and atomically."""
import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from app.backend.main import check_clip_name, create_app  # noqa: E402

REPORT = {"clip": "clip", "info": {"duration_s": 2.0, "width": 64, "height": 36, "fps": 24.0},
          "stages": ["shots"], "shots": [{"index": 1, "t0": 0.0, "t1": 2.0, "start": 0, "end": 47}]}


@pytest.fixture
def setup(tmp_path):
    clips, out = tmp_path / "clips", tmp_path / "out"
    clips.mkdir()
    (clips / "clip.mp4").write_bytes(b"\0" * 2048)
    (out / "clip").mkdir(parents=True)
    (out / "clip" / "report.json").write_text(json.dumps(REPORT), encoding="utf-8")
    # files next to and above the outputs folder that no URL may reach
    (tmp_path / "secret.json").write_text('{"secret": 1}', encoding="utf-8")
    (out / "report.json").write_text('{"secret": 2}', encoding="utf-8")
    app = create_app([clips], out)
    return TestClient(app, raise_server_exceptions=False), tmp_path, out


@pytest.mark.parametrize("name", ["..", ".", "", "a/b", "a\\b", "x\0y"])
def test_bad_clip_names_are_refused(name):
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        check_clip_name(name)


@pytest.mark.parametrize("name", ["clip", "My Film (2001)", "ep.01", "..dots", "film: part 2"])
def test_ordinary_clip_names_pass(name):
    assert check_clip_name(name) == name


@pytest.mark.parametrize("url", [
    "/api/clips/%2e%2e/report",
    "/api/clips/%2e%2e/files/secret.json",
    "/api/clips/%2e%2e/files/report.json",
    "/api/clips/%2e/report",
    "/api/clips/%2e%2e/export/html",
    "/api/clips/%2e%2e/export/edl",
    "/api/clips/%2e%2e/video",
    "/api/clips/%2e%2e/frame?t=0",
    "/api/clips/%2e%2e/campath",
    "/api/clips/%2e%2e/search?q=x",
    "/api/clips/%2e%2e/cutout?t=0",
    "/api/clips/clip/files/%2e%2e/%2e%2e/secret.json",
    "/api/compare?a=..&b=clip",
    "/api/compare?a=clip&b=..%2F..%2Fsecret",
])
def test_no_url_escapes_the_outputs_folder(setup, url):
    client, _, _ = setup
    r = client.get(url)
    assert r.status_code in (400, 404), (url, r.status_code)
    assert "secret" not in r.text or "not a clip name" in r.text


def test_valid_clip_still_works(setup):
    client, _, _ = setup
    assert client.get("/api/clips/clip/report").json()["clip"] == "clip"
    assert client.get("/api/clips/clip/files/report.json").status_code == 200
    assert client.get("/api/clips/clip/export/csv").status_code == 200


def test_upload_size_cap_leaves_nothing_behind(setup, monkeypatch):
    client, _, out = setup
    monkeypatch.setenv("CINESCOPE_MAX_UPLOAD_MB", "1")
    r = client.post("/api/clips/upload", files={"file": ("big.mp4", b"\0" * (3 << 20), "video/mp4")})
    assert r.status_code == 413
    assert "CINESCOPE_MAX_UPLOAD_MB" in r.json()["detail"]
    assert list((out / "_uploads").iterdir()) == []
    assert "big" not in {c["clip"] for c in client.get("/api/clips").json()}
    r = client.post("/api/clips/upload", files={"file": ("small.mp4", b"\0" * 4096, "video/mp4")})
    assert r.status_code == 200 and r.json()["clip"] == "small"
    assert [p.name for p in (out / "_uploads").iterdir()] == ["small.mp4"]


def test_failed_export_leaves_no_file(setup, monkeypatch):
    import cinescope.editorial.breakdown as bd

    client, _, out = setup

    def boom(rep, root, dest):
        Path(dest).write_text("<html>half", encoding="utf-8")
        raise RuntimeError("render crashed")

    monkeypatch.setattr(bd, "save_breakdown", boom)
    assert client.get("/api/clips/clip/export/html").status_code == 500
    exports = out / "clip" / "exports"
    assert not exports.exists() or list(exports.iterdir()) == []


def test_concurrent_exports_render_once(setup, monkeypatch):
    import cinescope.editorial.breakdown as bd

    client, _, out = setup
    calls = []

    def slow(rep, root, dest):
        calls.append(1)
        time.sleep(0.3)
        Path(dest).write_text("<html>done</html>", encoding="utf-8")
        return dest

    monkeypatch.setattr(bd, "save_breakdown", slow)
    results = []
    threads = [threading.Thread(target=lambda: results.append(client.get("/api/clips/clip/export/html")))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [r.status_code for r in results] == [200] * 4
    assert all(r.text == "<html>done</html>" for r in results)
    assert len(calls) == 1
    assert [p.name for p in (out / "clip" / "exports").iterdir()] == ["clip_breakdown.html"]


def test_finished_jobs_are_pruned(setup, monkeypatch):
    import app.backend.main as m

    monkeypatch.setattr(m, "MAX_FINISHED_JOBS", 3)
    clips_dir = setup[1] / "clips"
    app = create_app([clips_dir], setup[2], runner=lambda **kw: {"clip": "clip"})
    client = TestClient(app)
    for _ in range(6):
        j = client.post("/api/jobs", json={"clip": "clip", "stages": ["shots"]}).json()
        for _ in range(200):
            if client.get(f"/api/jobs/{j['id']}").json()["status"] not in ("queued", "running"):
                break
            time.sleep(0.01)
    assert len(app.state.jobs) <= 4                # 3 finished + the newest
