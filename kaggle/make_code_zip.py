"""Pack the code (no footage, no environment, no outputs) for upload as a Kaggle dataset.

    python kaggle/make_code_zip.py          # -> kaggle_upload/cinescope-code.zip (git-ignored)

Upload the zip at kaggle.com -> Datasets -> New Dataset (keep it private);
Kaggle unpacks it. After code changes, run this again and use
"New Version" on the same dataset.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["cinescope", "p0_classical", "p1_cnn", "p2_detection", "tests", "kaggle",
           "pyproject.toml", "requirements.txt", "README.md", "LICENSE"]
SKIP_DIRS = {"__pycache__", "checkpoints", "results", ".pytest_cache", ".ruff_cache"}


def main() -> Path:
    out = ROOT / "kaggle_upload" / "cinescope-code.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in INCLUDE:
            p = ROOT / name
            files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
            for f in files:
                rel = f.relative_to(ROOT)
                if SKIP_DIRS & set(rel.parts) or f.suffix in {".pt", ".pth", ".onnx", ".mkv", ".mp4"}:
                    continue
                z.write(f, Path("cinescope-code") / rel)
                n += 1
    print(f"{n} files -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    main()
