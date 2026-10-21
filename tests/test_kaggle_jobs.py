"""The Kaggle jobs assemble into valid scripts (preamble + job + any job it extends),
and every planned job has a source file. Nothing here talks to Kaggle."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run_all():
    spec = importlib.util.spec_from_file_location("run_all", ROOT / "kaggle" / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_planned_job_has_a_source(run_all):
    for name in run_all.PLAN:
        assert (run_all.JOBS / f"{name}.py").exists(), name
    assert set(run_all.DEFAULT) <= set(run_all.PLAN)


@pytest.mark.parametrize("name", sorted(p.stem for p in (ROOT / "kaggle" / "jobs").glob("p*.py")))
def test_job_assembles_to_valid_python(run_all, name):
    src = run_all.job_source(name)
    ast.parse(src)
    assert src.count("# --- end of preamble ---") == 1       # the base's preamble is not repeated


def test_extending_jobs_set_constants_before_the_base_reads_them(run_all):
    src = run_all.job_source("p2_train_v3")
    assert src.index('TAG, SIZE_MODE, EPOCHS = "centernet_r18_v3"') < src.index('TAG = globals().get("TAG"')
    assert '"--keep-large", "64"' in src
    src = run_all.job_source("p9_tensorrt_v2")
    assert src.index('OUT_TAG = "tensorrt_v2"') < src.index("OUT = f\"/kaggle/working/results/")


def test_unknown_base_is_an_error(run_all, tmp_path, monkeypatch):
    (tmp_path / "_preamble.py").write_text("# --- end of preamble ---\n")
    (tmp_path / "pX.py").write_text("# extends: nope\n")
    monkeypatch.setattr(run_all, "JOBS", tmp_path)
    with pytest.raises(SystemExit, match="nope"):
        run_all.job_source("pX")
