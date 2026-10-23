"""Is this machine ready to run CineScope? One command answers, with the fix for each gap.

    python -m cinescope.doctor            # or: cinescope --doctor
    python -m cinescope.doctor --json     # the same as JSON (the web app's /api/doctor)
    python -m cinescope.doctor --verify   # also SHA-256 every weight (reads ~700 MB)

It checks the Python packages, FFmpeg's codecs (through PyAV), the model weights
against cinescope.assets, free disk space, and then says, stage by stage, which
pipeline stages can run and what to do about the rest. A stage that cannot run
does not stop a pipeline run (its error is reported and the others finish), but
finding out here takes a second instead of a failed stage an hour into a film.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path

from .assets import REPO, WEIGHTS, check, weights_dir

# module -> (what it is for, how to install it)
PACKAGES = {
    "numpy": ("everything", "pip install -r requirements.txt"),
    "av": ("video decoding (PyAV / FFmpeg)", "pip install av"),
    "cv2": ("faces, camera paths, baselines", "pip install opencv-python"),
    "PIL": ("images and exports", "pip install pillow"),
    "torch": ("the deep-learning stages", "pip install torch torchvision"),
    "onnxruntime": ("the ONNX backend (web app default)", "pip install onnxruntime"),
    "open_clip": ("scene search and scene types", "pip install open_clip_torch"),
    "ultralytics": ("pose and cut-outs", "pip install ultralytics"),
    "rapidocr_onnxruntime": ("subtitle OCR without a CRNN checkpoint", "pip install rapidocr-onnxruntime"),
    "fastapi": ("the web app", "pip install fastapi uvicorn python-multipart"),
    "opentimelineio": ("OTIO export", "pip install opentimelineio"),
}

FETCH = "python tools/fetch_assets.py --only weights"


def have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _need_ok(need: str, weight_rows: dict) -> tuple[bool, str]:
    """(met, fix) for one "kind:what" need; "a|b" is met by either."""
    fixes = []
    for alt in need.split("|"):
        kind, _, what = alt.partition(":")
        if kind == "py":
            if have(what):
                return True, ""
            fixes.append(PACKAGES.get(what, ("", f"pip install {what}"))[1])
        elif kind == "weight":
            row = weight_rows.get(what) or {"status": "missing"}
            if row["status"] in ("ok", "unverified"):
                return True, ""
            fixes.append(f"{FETCH}  ({what} is {row['status']})")
        elif kind == "file":
            if (REPO / what).exists():
                return True, ""
            fixes.append(f"{what} (train or copy it)")
    return False, " or ".join(fixes)


def codecs() -> dict:
    """What FFmpeg (as PyAV built it) can do that CineScope relies on."""
    if not have("av"):
        return {}
    import av

    def enc(name: str) -> bool:
        try:
            av.codec.Codec(name, "w")
            return True
        except Exception:                       # noqa: BLE001 - absent codec
            return False

    def dec(name: str) -> bool:
        try:
            av.codec.Codec(name, "r")
            return True
        except Exception:                       # noqa: BLE001
            return False

    return {"ffmpeg": getattr(av, "ffmpeg_version_info", None) or "?",
            "decode": {c: dec(c) for c in ("h264", "hevc", "vp9", "av1", "mpeg4", "prores")},
            "encode": {"libx264 (highlight reels)": enc("libx264"), "aac (reel audio)": enc("aac")}}


def diagnose(out: str | Path = "outputs", verify: bool = False) -> dict:
    from .pipeline import STAGES
    from .pipeline import stages as _builtin  # noqa: F401  (registers the stages)

    packages = {m: {"ok": have(m), "for": why, "fix": fix} for m, (why, fix) in PACKAGES.items()}
    weights = {w.name: check(w, hash_it=verify) for w in WEIGHTS.values()}
    stages = {}
    for name, cls in STAGES.items():
        missing = []
        for need in getattr(cls, "needs", ()):
            ok, fix = _need_ok(need, weights)
            if not ok:
                missing.append({"need": need, "fix": fix})
        stages[name] = {"ready": not missing, "missing": missing, "default": getattr(cls, "default", True),
                        "requires": list(cls.requires)}
    # a stage is only as ready as the stages it needs
    for name in STAGES:
        blocked = [d for d in STAGES[name].requires if not stages[d]["ready"]]
        if blocked and stages[name]["ready"]:
            stages[name] = {**stages[name], "ready": False,
                            "missing": [{"need": f"stage:{d}", "fix": f"fix {d} first"} for d in blocked]}
    o = Path(out)
    probe = o if o.exists() else o.resolve().parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    from . import __version__

    return {"version": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
            "weights_dir": str(weights_dir()), "packages": packages, "codecs": codecs(), "weights": weights,
            "stages": stages, "disk": {"path": str(probe), "free_gb": round(free / 2**30, 1),
                                       "low": free < 5 * 2**30},
            "ready": [n for n, s in stages.items() if s["ready"]],
            "not_ready": [n for n, s in stages.items() if not s["ready"]]}


def render(d: dict) -> str:
    ok, no = "ok ", "-- "
    lines = [f"CineScope {d['version']} · Python {d['python']} · {d['platform']}", "", "Packages"]
    for m, p in d["packages"].items():
        lines.append(f"  {ok if p['ok'] else no}{m:<22} {p['for']}" + ("" if p["ok"] else f"   -> {p['fix']}"))
    c = d["codecs"]
    if c:
        lines += ["", f"FFmpeg {c['ffmpeg']}"]
        lines.append("  decode: " + ", ".join(f"{k}{'' if v else ' (missing)'}" for k, v in c["decode"].items()))
        lines.append("  encode: " + ", ".join(f"{k}{'' if v else ' (missing)'}" for k, v in c["encode"].items()))
    lines += ["", f"Weights in {d['weights_dir']}"]
    for w in d["weights"].values():
        mark = ok if w["status"] in ("ok", "unverified") else no
        lines.append(f"  {mark}{w['name']:<38} {w['status']:<10} {w['detail']}")
    lines += ["", "Stages"]
    for n, s in d["stages"].items():
        if s["ready"]:
            lines.append(f"  {ok}{n}")
        else:
            lines.append(f"  {no}{n:<12} " + "; ".join(m["fix"] for m in s["missing"]))
    dk = d["disk"]
    lines += ["", f"Disk: {dk['free_gb']} GB free at {dk['path']}" + ("  (low: a film's results take GBs)"
                                                                     if dk["low"] else "")]
    nr = d["not_ready"]
    lines += ["", f"{len(d['ready'])} of {len(d['stages'])} stages ready."
              + (f" Not ready: {', '.join(nr)}." if nr else " Everything is ready.")]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cinescope.doctor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verify", action="store_true", help="SHA-256 every weight file")
    ap.add_argument("--out", default="outputs", help="where results go (for the disk check)")
    args = ap.parse_args(argv)
    d = diagnose(args.out, verify=args.verify)
    print(json.dumps(d, indent=1) if args.json else render(d))
    bad = [w for w in d["weights"].values() if w["status"] == "bad"]
    return 1 if bad or not d["packages"]["av"]["ok"] and not d["packages"]["cv2"]["ok"] else 0


if __name__ == "__main__":
    sys.exit(main())
