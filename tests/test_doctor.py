"""cinescope.doctor and the weights table: what can run here, and downloads that verify."""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from cinescope.assets import WEIGHTS, Weight, check
from cinescope.doctor import diagnose, main, render
from cinescope.pipeline import STAGES
from cinescope.pipeline import stages as _builtin  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def _fetch_assets():
    spec = importlib.util.spec_from_file_location("fetch_assets", ROOT / "tools" / "fetch_assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_weight_a_stage_needs_is_in_the_table():
    for name, cls in STAGES.items():
        for need in cls.needs:
            for alt in need.split("|"):
                kind, _, what = alt.partition(":")
                assert kind in ("py", "weight", "file"), (name, need)
                if kind == "weight":
                    assert what in WEIGHTS, (name, what)


def test_cutouts_weight_is_fetched():
    assert "yolov8n-seg.pt" in WEIGHTS and "cutouts" in WEIGHTS["yolov8n-seg.pt"].used_by
    assert all(w.urls and all(u.startswith("https://") for u in w.urls) for w in WEIGHTS.values())


def test_missing_weights_name_the_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESCOPE_WEIGHTS", str(tmp_path))
    d = diagnose(tmp_path)
    assert set(d["stages"]) == set(STAGES)
    assert d["stages"]["shots"]["ready"] and "shots" in d["ready"]
    faces = d["stages"]["faces"]
    assert not faces["ready"]
    assert any("fetch_assets.py" in m["fix"] and "yunet.onnx" in m["fix"] for m in faces["missing"])
    # a stage that needs a stage that cannot run cannot run either
    assert not d["stages"]["speaking"]["ready"]
    text = render(d)
    assert "stages ready" in text and "faces" in text


def test_present_weight_makes_the_stage_ready(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.setenv("CINESCOPE_WEIGHTS", str(tmp_path))
    (tmp_path / "yunet.onnx").write_bytes(b"\1" * 4096)
    d = diagnose(tmp_path)                                    # no hashing by default: fast
    assert d["weights"]["yunet.onnx"]["status"] == "ok"
    assert d["stages"]["faces"]["ready"]
    d = diagnose(tmp_path, verify=True)                       # the bytes are not YuNet's
    assert d["weights"]["yunet.onnx"]["status"] == "bad"
    assert not d["stages"]["faces"]["ready"]


@pytest.mark.parametrize("content, why", [
    (b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n", "LFS"),
    (b"<!DOCTYPE html><html>rate limited</html>", "HTML"),
])
def test_bad_downloads_are_recognised(tmp_path, content, why):
    (tmp_path / "x.onnx").write_bytes(content)
    r = check(Weight("x.onnx", ("https://example.invalid/x",), None, ()), tmp_path)
    assert r["status"] == "bad" and why in r["detail"]


def test_checksum(tmp_path):
    data = b"\7" * 5000
    (tmp_path / "w.bin").write_bytes(data)
    good = Weight("w.bin", ("https://x",), hashlib.sha256(data).hexdigest(), ())
    assert check(good, tmp_path)["status"] == "ok"
    assert check(Weight("w.bin", ("https://x",), "0" * 64, ()), tmp_path)["status"] == "bad"
    assert check(Weight("w.bin", ("https://x",), None, ()), tmp_path)["status"] == "unverified"


def test_fetch_tries_mirrors_and_verifies(tmp_path):
    fa = _fetch_assets()
    src = tmp_path / "src"
    src.mkdir()
    data = b"\5" * 8192
    (src / "good.bin").write_bytes(data)
    (src / "wrong.bin").write_bytes(b"\6" * 8192)
    dest = tmp_path / "weights"
    w = Weight("model.bin", ((src / "missing.bin").as_uri(), (src / "wrong.bin").as_uri(), (src / "good.bin").as_uri()),
               hashlib.sha256(data).hexdigest(), ())
    assert fa.fetch_weight(w, dest) is True
    assert (dest / "model.bin").read_bytes() == data
    assert [p.name for p in dest.iterdir()] == ["model.bin"]              # no .part left behind
    assert fa.fetch_weight(w, dest) is False                               # present: skipped
    only_wrong = Weight("other.bin", ((src / "wrong.bin").as_uri(),), w.sha256, ())
    with pytest.raises(RuntimeError, match="SHA-256"):
        fa.fetch_weight(only_wrong, dest)
    assert not (dest / "other.bin").exists() and not (dest / "other.bin.part").exists()


def test_cli(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("CINESCOPE_WEIGHTS", str(tmp_path))
    assert main(["--out", str(tmp_path)]) in (0, 1)
    assert "stages ready" in capsys.readouterr().out
    main(["--json", "--out", str(tmp_path)])
    import json

    assert "stages" in json.loads(capsys.readouterr().out)
    from cinescope.pipeline.__main__ import main as pipeline_main

    pipeline_main(["--doctor", "--out", str(tmp_path)])
    assert "Stages" in capsys.readouterr().out


def test_api_doctor(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from app.backend.main import create_app

    c = TestClient(create_app([tmp_path], tmp_path / "out"))
    d = c.get("/api/doctor").json()
    assert d["stages"]["shots"]["ready"] and "not_ready" in d


def test_fetch_assets_list_shows_mirrors(capsys):
    fa = _fetch_assets()
    assert fa.main(["--list", "--only", "weights"]) == 0
    out = capsys.readouterr().out
    assert "yolov8n-seg.pt" in out and "libfacedetection" in out
    assert sys.modules.get("cinescope.assets") is not None
