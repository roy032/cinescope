"""Chat with the movie: ask a question, get an answer that cites timestamps.

    python p8_search/chat.py trailers/goldeneye.mp4 "when does the car chase start?"
    python p8_search/chat.py trailers/goldeneye.mp4 "is there a plane?" --json
    python p8_search/chat.py clip.mp4 "what happens in the snow?" --srt clip.srt     # subtitles as evidence
    python p8_search/chat.py clip.mp4                                                 # interactive

Retrieves the best-matching segments (search.py's index, built on first use), shows
the evidence (times, zero-shot tags, subtitles/characters if attached) and answers:

  * default: an extractive template answer from the evidence (no network, no key);
  * with ANTHROPIC_API_KEY and CINESCOPE_LLM_MODEL set (and `pip install anthropic`):
    the evidence and key frames go to the model, which must answer from them only.
    --no-llm forces the template.
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
from cinescope.search.rag import RagParams, answer, llm_available  # noqa: E402
from cinescope.search.shotindex import attach_subtitles, parse_srt  # noqa: E402


def show(ans, as_json: bool) -> None:
    if as_json:
        print(json.dumps(ans.to_json(), indent=1))
        return
    print(f"\nQ: {ans.question}\n   (searched for: \"{ans.query}\", question type: {ans.kind})")
    print("Evidence:")
    for e in ans.evidence:
        d = e.doc
        tags = ", ".join(t["tag"] for t in d.get("tags", [])[:3])
        extra = ""
        if d.get("subtitles"):
            extra += "  subs: " + " / ".join(s["text"] for s in d["subtitles"])[:80]
        if d.get("characters"):
            extra += "  who: " + ", ".join(c["name"] for c in d["characters"])
        print(f"  {e.span():>14s}  shot {d['shot']:3d}  sim {e.clip_score:.3f} ({e.contrast:+.3f} vs median)"
              f"  [{tags}]{extra}")
    print(f"A ({ans.source}): {ans.text}")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--encoder", default="openclip")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--srt", help="subtitle file to attach as evidence")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    enc = get_encoder(args.encoder)
    ix = open_index(args.video, enc, args.out)
    if args.srt:
        attach_subtitles(ix.docs, parse_srt(Path(args.srt).read_text(encoding="utf-8", errors="replace")))
    use_llm = False if args.no_llm else llm_available()
    if not args.json:
        print(f"{len(ix)} segments indexed; answers by {'LLM' if use_llm else 'template (no LLM configured)'}")
    p = RagParams(k=args.k)
    if args.question:
        ans = answer(args.question, ix, enc, p, use_llm=use_llm)
        show(ans, args.json)
        return ans
    while True:                                          # interactive
        try:
            q = input("\nask> ").strip()
        except EOFError:
            return None
        if q.lower() in {"", "q", "quit", "exit"}:
            return None
        show(answer(q, ix, enc, p, use_llm=use_llm), args.json)


if __name__ == "__main__":
    main()
