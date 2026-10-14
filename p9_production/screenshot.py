"""Screenshots of the web app with headless Chromium (Playwright).

    uvicorn app.backend.main:app --port 8000 &            # with app/frontend/dist built
    python p9_production/screenshot.py goldeneye --query "a plane in the snow"

-> p9_production/results/app_<clip>.jpg (whole page) and app_<clip>_search.jpg
(the top of the page after a scene search), JPEG-compressed to stay small.

The Chromium binary is found under $PLAYWRIGHT_BROWSERS_PATH (or
/opt/pw-browsers); pass --chrome to point at another one.
"""
from __future__ import annotations

import argparse
import glob
import io
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_chrome() -> str | None:
    base = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
    hits = sorted(glob.glob(f"{base}/chromium*/chrome-linux*/chrome"))
    return hits[-1] if hits else None


def save_jpeg(png: bytes, path: Path, max_kb: int = 290, width: int | None = None) -> None:
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    for q in (85, 78, 70, 62, 55, 48, 40):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, optimize=True, progressive=True)
        if buf.tell() <= max_kb * 1024:
            break
    path.write_bytes(buf.getvalue())
    print(f"{path} {buf.tell() // 1024} KB (q={q})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--query", default="a man walks away from an explosion")
    ap.add_argument("--seek", type=float, default=None, help="click the barcode at this fraction (0..1)")
    ap.add_argument("--chrome", default=find_chrome())
    ap.add_argument("--out", default=str(ROOT / "p9_production" / "results"))
    args = ap.parse_args(argv)

    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chrome, headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1)
        page.goto(f"{args.url}/?clip={args.clip}", wait_until="networkidle")
        page.wait_for_selector(".shotstrip img", timeout=30000)
        if args.seek is not None:
            bc = page.locator(".barcode")
            box = bc.bounding_box()
            page.mouse.click(box["x"] + args.seek * box["width"], box["y"] + box["height"] / 2)
            page.wait_for_timeout(1500)
        page.wait_for_timeout(1000)
        save_jpeg(page.screenshot(full_page=True), out / f"app_{args.clip}.jpg", width=1200)
        page.fill("input[aria-label='search query']", args.query)
        page.click(".search button.primary")
        page.wait_for_selector(".hits .hit img", timeout=60000)
        page.wait_for_timeout(1500)
        save_jpeg(page.screenshot(full_page=False), out / f"app_{args.clip}_search.jpg", width=1200)
        browser.close()


if __name__ == "__main__":
    main()
