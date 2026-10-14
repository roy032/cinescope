"""Run a fixed list of chat questions on public clips and save the answers (results/chat_examples.json).

    python p8_search/chat_examples.py --trailers trailers

The questions were written after the search evaluation; they illustrate the answer
types ("when", yes/no, "what", "who", the "not found" case) and are not scored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "p8_search"))

from search import open_index  # noqa: E402

from cinescope.search.encoders import get_encoder  # noqa: E402
from cinescope.search.rag import answer  # noqa: E402

QUESTIONS = {
    "goldeneye.mp4": ["When does the car chase start?",
                      "Is there an explosion?",
                      "When does someone light a cigarette?",
                      "Is there a plane?",
                      "What happens in the snow?",
                      "Who is at the computer?"],
    "people-detection.mp4": ["When do three people walk in together?",
                             "How many times is the room empty?"],
    "Megamind.avi": ["Where are they having dinner?"],
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trailers", default=str(ROOT / "trailers"))
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)
    enc = get_encoder("openclip")
    out = []
    for clip, qs in QUESTIONS.items():
        ix = open_index(str(Path(args.trailers) / clip), enc, args.out)
        for q in qs:
            a = answer(q, ix, enc, use_llm=False)
            print(f"[{clip}] Q: {q}\n  A: {a.text}\n")
            out.append({"clip": clip, **a.to_json()})
    path = ROOT / "p8_search" / "results" / "chat_examples.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
