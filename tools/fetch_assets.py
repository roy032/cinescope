"""Download the public test clips and the library weights every phase reads.

    python tools/fetch_assets.py                    # clips -> trailers/, weights -> data/weights/
    python tools/fetch_assets.py --only clips       # or: weights
    python tools/fetch_assets.py --list             # what would be fetched, and from where
    python tools/fetch_assets.py --check            # verify the weights already present (SHA-256)

Everything comes from the projects' own public releases (the same URLs the phase
READMEs cite). Files already present are skipped, so it is safe to re-run. Nothing
here is committed: trailers/ and data/ are git-ignored (README: Footage and copyright).
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_SCENEDETECT = "https://raw.githubusercontent.com/Breakthrough/PySceneDetect/resources/tests/resources"
_INTEL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master"
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

# Public clips the phases are scored on: Phase 0 cuts, Phase 2 week 8, Phases 3-9.
CLIPS = {
    "goldeneye.mp4": f"{_SCENEDETECT}/goldeneye.mp4",
    "testvideo.mp4": f"{_SCENEDETECT}/testvideo.mp4",
    "head-pose-face-detection-female-and-male.mp4": f"{_INTEL}/head-pose-face-detection-female-and-male.mp4",
    "classroom.mp4": f"{_INTEL}/classroom.mp4",
}

# Library weights: the table lives in cinescope/assets.py (mirrors, SHA-256, which stage reads each),
# shared with `python -m cinescope.doctor`, which checks the files against it.
sys.path.insert(0, str(ROOT))
from cinescope.assets import WEIGHTS, Weight, check, looks_wrong, sha256, weights_dir  # noqa: E402


def fetch(url: str, dest: Path, expect_sha256: str | None = None) -> bool:
    """Download url -> dest unless it exists. A partial or wrong download never lands under the final name."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "cinescope-fetch"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        bad = looks_wrong(tmp)
        if bad:
            raise RuntimeError(f"{url}: {bad}")
        if expect_sha256 and (got := sha256(tmp)) != expect_sha256:
            raise RuntimeError(f"{url}: SHA-256 {got[:12]}… does not match {expect_sha256[:12]}…")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return True


def fetch_weight(w: Weight, folder: Path) -> bool:
    """Try each mirror in turn; the first that gives the right file wins."""
    errors = []
    for url in w.urls:
        try:
            return fetch(url, folder / w.name, w.sha256)
        except Exception as e:                    # noqa: BLE001 - next mirror
            errors.append(str(e))
    raise RuntimeError("; ".join(errors))


def plan(only: str | None = None) -> list[tuple[str, Path]]:
    jobs = []
    if only in (None, "clips"):
        jobs += [(u, ROOT / "trailers" / n) for n, u in CLIPS.items()]
    if only in (None, "weights"):
        jobs += [(w.urls[0], weights_dir() / n) for n, w in WEIGHTS.items()]
    return jobs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["clips", "weights"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true", help="verify the weights already present (SHA-256)")
    args = ap.parse_args(argv)
    if args.check:
        rows = [check(w) for w in WEIGHTS.values()]
        for r in rows:
            print(f"{r['status']:<10} {r['name']:<38} {r['detail']}")
        return 1 if any(r["status"] == "bad" for r in rows) else 0
    failed = 0
    for url, dest in plan(args.only):
        rel = dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest
        w = WEIGHTS.get(dest.name) if dest.parent == weights_dir() else None
        if args.list:
            print(f"{rel}  <-  {'  |  '.join(w.urls) if w else url}")
            continue
        try:
            got = fetch_weight(w, dest.parent) if w else fetch(url, dest)
            print(f"{'fetched' if got else 'present'}  {rel}")
        except Exception as e:                    # keep going: one dead mirror should not block the rest
            failed += 1
            print(f"FAILED   {rel}: {e}", file=sys.stderr)
    if not args.only or args.only == "weights":
        bad = [r for r in (check(w, hash_it=False) for w in WEIGHTS.values()) if r["status"] == "bad"]
        for r in bad:
            print(f"BAD      {r['name']}: {r['detail']} (delete it and run again)", file=sys.stderr)
        failed += len(bad)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
