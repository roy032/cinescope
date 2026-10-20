"""Packaging: the console scripts resolve, versions agree, and command-line tools answer --help
without doing their work."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import cinescope

ROOT = Path(__file__).resolve().parents[1]
tomllib = pytest.importorskip("tomllib")   # Python 3.11+


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_console_scripts_resolve():
    for name, target in _pyproject()["project"]["scripts"].items():
        module, func = target.split(":")
        assert callable(getattr(importlib.import_module(module), func)), name


def test_versions_agree():
    version = _pyproject()["project"]["version"]
    assert cinescope.__version__ == version
    front = json.loads((ROOT / "app/frontend/package.json").read_text(encoding="utf-8"))
    assert front["version"] == version


@pytest.mark.parametrize("script", ["kaggle/make_code_zip.py", "tools/update_results.py"])
def test_help_does_no_work(script, tmp_path):
    before = sorted(p.name for p in ROOT.iterdir())
    r = subprocess.run([sys.executable, str(ROOT / script), "--help"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.startswith("usage:")
    assert sorted(p.name for p in ROOT.iterdir()) == before
