"""Chat with the movie: retrieval-augmented answers grounded in shots and timestamps.

    question ──► search query ──► retrieve segments ──► evidence ──► answer
                 (strip "when      (CLIP similarity     (times,       template (default) or
                  does ... start")  + BM25 over tags,     tags, key     an LLM given the same
                                    subtitles, names)    frames)       evidence (optional)

Why retrieve at all? A 2-hour film is ~170k frames; no model reads all of them per
question. Retrieval picks a handful of segments, and the answer may only use what
they contain, so every claim carries a timestamp a viewer can check.

Hybrid retrieval. The CLIP score sees pixels; the lexical score (BM25, from
scratch below) sees words attached to a segment — zero-shot tags today, and the
`subtitles` / `characters` fields when Phase 7 (OCR) and Phase 3 (faces) fill
them. "When does Bond say 'for England'?" is a subtitle question no image
embedding can answer; "the snowy satellite dish" is the opposite.

Knowing when not to answer. CLIP similarities are not probabilities: the best
segment for "a car chase" in a clip without one still scores ~0.2. Below
`min_score` the answer says no segment matches. The threshold (0.219, for the
open_clip ViT-B/32 LAION-400M encoder) was chosen on the development clip
(vtest.avi) only, halfway between its weakest present query (0.223) and its
strongest absent one ("a car chase", 0.215 — there are parked cars in the shot):
calibrate_chat.py. The *contrast* (best minus the median over all segments) is
reported too, but it did not separate present from absent on that clip — a static
camera makes every segment look alike, so a query that matches matches everywhere.

The LLM is optional. With ANTHROPIC_API_KEY and CINESCOPE_LLM_MODEL set and the
`anthropic` package installed, the evidence (and key frames) go to the model with
instructions to answer only from them and cite timestamps; otherwise — the
default — an extractive template answer is written from the same evidence.
"""
from __future__ import annotations

import base64
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Generic visual concepts for zero-shot tags (evidence for the answer, not for scoring).
TAG_VOCAB = [
    # people
    "a man", "a woman", "a child", "two people", "a group of people", "a crowd", "a face in close-up",
    "a person walking", "a person running", "a person sitting", "a person standing", "people talking",
    "people dancing", "a person smoking", "a person shouting", "a person smiling", "a person with glasses",
    "a person from behind", "a soldier", "a police officer", "an animated character",
    # actions and events
    "an explosion", "fire", "smoke", "a gunfight", "a fight", "a car chase", "a car crash", "a kiss",
    "a hug", "a phone call", "people eating", "someone reading", "someone typing on a computer",
    "a celebration", "a chase on foot",
    # objects
    "a car", "a truck", "a motorcycle", "a bicycle", "a plane", "a helicopter", "a boat", "a train",
    "a gun", "a knife", "a computer screen", "a map", "a logo", "text on screen", "a phone", "a television",
    "a lighter", "a cigarette", "a glass of wine", "candles", "a table", "a chair", "a door", "a window",
    "a book", "a satellite dish", "a desk", "a whiteboard",
    # places
    "an office", "a control room", "a classroom", "a corridor", "an empty room", "a kitchen",
    "a bedroom", "a restaurant", "a bar", "a street", "a city at night", "a road", "a forest", "a field",
    "a park", "a beach", "the sea", "mountains", "snow", "a desert", "a river", "the sky", "a building",
    "a parking lot", "a factory", "a shop", "a church", "a stage",
    # look
    "a dark scene", "a night scene", "a bright daylight scene", "a black screen", "a blurry image",
    "a close-up shot", "a wide shot", "an aerial view", "a cartoon", "black and white footage",
    # nature
    "a deer", "a dog", "a cat", "a horse", "a bird", "leaves", "a rock", "tree bark", "flowers", "water",
]


def zero_shot_tags(vectors: np.ndarray, encoder, vocab: list[str], n: int = 5,
                   template: str = "a film still of {}.") -> list[list[dict]]:
    """Top-n vocabulary concepts per vector, by CLIP similarity with a prompt template."""
    t = encoder.encode_texts([template.format(v) for v in vocab])
    s = vectors @ t.T
    out = []
    for row in s:
        top = np.argsort(-row)[:n]
        out.append([{"tag": vocab[i], "score": round(float(row[i]), 4)} for i in top])
    return out


# --------------------------------------------------------------------------- lexical retrieval
_WORD = re.compile(r"[a-z0-9']+")
_STOP_WORDS = """
    a an the of in on at to is are was were be been does do did when where who what which how there
    this that it its and or for with from by as into out up down any some scene shot moment part
    clip film movie video happen happens start starts begin begins show shows appear appears first
    last time
"""
STOP = frozenset(_STOP_WORDS.split())


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOP]


def doc_text(d: dict) -> str:
    parts = [t["tag"] for t in d.get("tags", [])]
    parts += [s["text"] for s in d.get("subtitles", [])]
    parts += [c["name"] for c in d.get("characters", [])]
    return " ".join(parts)


class BM25:
    """Okapi BM25 (Robertson & Zaragoza, 2009): idf-weighted term frequency, saturated
    by k1 and normalised by document length relative to the average (b)."""

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.toks = [tokens(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = np.mean([len(t) for t in self.toks]) if self.toks else 0.0
        df = Counter(w for t in self.toks for w in set(t))
        n = len(self.toks)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def scores(self, query: str) -> np.ndarray:
        q = tokens(query)
        out = np.zeros(len(self.toks))
        for i, (tf, t) in enumerate(zip(self.tf, self.toks, strict=True)):
            norm = self.k1 * (1 - self.b + self.b * len(t) / max(self.avg, 1e-9))
            out[i] = sum(self.idf.get(w, 0.0) * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in q if w in tf)
        return out


# --------------------------------------------------------------------------- questions
_LEAD = re.compile(r"^(when|where|at what (point|time)|what time|is there|are there|does|do|did|who|what|"
                   r"how many times|how many|how|which|show me|find|can you find)\b\s*", re.I)
_AUX = re.compile(r"^(is|are|does|do|did|was|were|the scene where|the part where|the moment when)\b\s*", re.I)
_TAIL = re.compile(r"\s*\b(start|starts|begin|begins|happen|happens|appear|appears|show up|shows up|occur|"
                   r"occurs|take place|in (the|this) (clip|film|movie|video)|on screen)\s*$", re.I)


def question_kind(q: str) -> str:
    ql = q.lower().strip()
    if re.match(r"^(when|at what (point|time)|what time)\b", ql) or re.search(r"\b(start|begin)s?\b", ql):
        return "when"
    if ql.startswith("who"):
        return "who"
    if ql.startswith("how many"):
        return "count"
    if re.match(r"^(is|are|does|do|did|was|were)\b", ql):
        return "yesno"
    if ql.startswith("where"):
        return "where"
    return "what"


def question_to_query(q: str) -> str:
    """'When does the car chase start?' -> 'the car chase' (what the shot should look like)."""
    s = q.strip().rstrip("?.! ")
    for _ in range(3):
        s2 = _AUX.sub("", _LEAD.sub("", s)).strip()
        s2 = _TAIL.sub("", s2).strip()
        if s2 == s:
            break
        s = s2
    return s or q


# --------------------------------------------------------------------------- retrieval + evidence
@dataclass
class RagParams:
    k: int = 5
    lexical_weight: float = 0.02      # BM25 is ~0-10, CLIP cosine ~0.1-0.35: a few hundredths per BM25 unit
    min_score: float = 0.219          # best CLIP cosine below this -> "no match" (calibrate_chat.py, dev clip)
    group_margin: float = 0.015       # segments within this of the best count as "the same answer"


@dataclass
class Evidence:
    doc: dict
    clip_score: float
    lexical: float
    score: float
    contrast: float

    def span(self) -> str:
        return f"{fmt_t(self.doc['t0'])}–{fmt_t(self.doc['t1'])}"


@dataclass
class Answer:
    question: str
    query: str
    kind: str
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    confident: bool = False
    source: str = "template"

    def to_json(self) -> dict:
        return {"question": self.question, "query": self.query, "kind": self.kind, "answer": self.text,
                "confident": self.confident, "source": self.source,
                "evidence": [{"clip": e.doc.get("clip"), "shot": e.doc["shot"], "part": e.doc.get("part", 0),
                              "t0": e.doc["t0"], "t1": e.doc["t1"], "clip_score": round(e.clip_score, 4),
                              "lexical": round(e.lexical, 3), "contrast": round(e.contrast, 4),
                              "tags": [t["tag"] for t in e.doc.get("tags", [])][:5],
                              "subtitles": [s["text"] for s in e.doc.get("subtitles", [])],
                              "characters": [c["name"] for c in e.doc.get("characters", [])]}
                             for e in self.evidence]}


def fmt_t(t: float) -> str:
    m, s = divmod(max(0.0, float(t)), 60)
    return f"{int(m)}:{s:04.1f}"


def retrieve(question: str, index, encoder, p: RagParams | None = None) -> tuple[str, list[Evidence]]:
    p = p or RagParams()
    query = question_to_query(question)
    q = encoder.encode_texts([query])[0]
    clip_scores = index.vectors @ q                     # exact: a clip has hundreds of segments, not millions
    lex = BM25([doc_text(d) for d in index.docs]).scores(query)
    total = clip_scores + p.lexical_weight * lex
    med = float(np.median(clip_scores))
    order = np.argsort(-total)[: p.k]
    return query, [Evidence(index.docs[i], float(clip_scores[i]), float(lex[i]), float(total[i]),
                            float(clip_scores[i]) - med) for i in order]


def _group(ev: list[Evidence], margin: float) -> list[Evidence]:
    """The segments that are nearly as good as the best one, in time order."""
    best = ev[0].score
    return sorted([e for e in ev if e.score >= best - margin], key=lambda e: (e.doc.get("clip", ""), e.doc["t0"]))


def _merge_spans(ev: list[Evidence], gap: float = 0.5) -> list[tuple[float, float]]:
    spans: list[list[float]] = []
    for e in ev:
        if spans and e.doc["t0"] <= spans[-1][1] + gap:
            spans[-1][1] = max(spans[-1][1], e.doc["t1"])
        else:
            spans.append([e.doc["t0"], e.doc["t1"]])
    return [(a, b) for a, b in spans]


def template_answer(question: str, query: str, ev: list[Evidence], p: RagParams | None = None) -> Answer:
    p = p or RagParams()
    kind = question_kind(question)
    if not ev:
        return Answer(question, query, kind, "The index is empty.", [], False)
    # Tags are CLIP's own guesses, so a tag-word hit is no independent evidence; a subtitle or
    # character-name hit is.
    said = bool(set(tokens(query)) & set(tokens(" ".join(s["text"] for s in ev[0].doc.get("subtitles", [])) + " "
                                               + " ".join(c["name"] for c in ev[0].doc.get("characters", [])))))
    confident = ev[0].clip_score >= p.min_score or said
    best = ev[0]
    tags = ", ".join(t["tag"] for t in best.doc.get("tags", [])[:3])
    where = f"shot {best.doc['shot']}" + (f" part {best.doc['part'] + 1}/{best.doc['n_parts']}"
                                          if best.doc.get("n_parts", 1) > 1 else "")
    if not confident:
        text = (f"I could not find \"{query}\" in this clip: no segment stands out (best match {best.span()}, "
                f"{where}, similarity {best.clip_score:.3f} < {p.min_score:.3f}, {best.contrast:+.3f} vs the clip's median). "
                f"Closest-looking segment: {tags or 'no tags'}.")
        return Answer(question, query, kind, text, ev, False)
    group = _group(ev, p.group_margin)
    spans = _merge_spans(group)
    span_txt = ", ".join(f"{fmt_t(a)}–{fmt_t(b)}" for a, b in spans)
    subs = [s["text"] for e in group for s in e.doc.get("subtitles", [])][:3]
    chars = sorted({c["name"] for e in group for c in e.doc.get("characters", [])})
    if kind == "when":
        first = spans[0]
        text = f"\"{query}\" first appears at {fmt_t(first[0])} ({fmt_t(first[0])}–{fmt_t(first[1])})."
        if (best.doc["t0"], best.doc["t1"]) != first and not (first[0] <= best.doc["t0"] < first[1]):
            text += f" The strongest match is later, {best.span()} ({where})."
        if len(spans) > 1:
            text += f" Similar moments: {', '.join(f'{fmt_t(a)}–{fmt_t(b)}' for a, b in spans[1:])}."
    elif kind == "count":
        text = f"{len(spans)} separate stretch{'es' if len(spans) != 1 else ''} match \"{query}\": {span_txt}."
    elif kind == "yesno":
        text = f"Yes, most likely: \"{query}\" matches {span_txt} (best {best.span()}, {where})."
    elif kind == "who":
        text = (f"On screen at {best.span()}: {', '.join(chars)}." if chars else
                f"The best match is {best.span()} ({where}); no character names are attached to this index "
                f"yet (Phase 3 fills them). The frames look like: {tags}.")
    else:
        text = f"Best match: {best.span()} ({where}), which looks like {tags}."
        if len(spans) > 1:
            others = [(a, b) for a, b in spans if not (a <= best.doc['t0'] < b)]
            if others:
                text += f" Other matching moments: {', '.join(f'{fmt_t(a)}–{fmt_t(b)}' for a, b in others)}."
    if subs:
        text += " Dialogue there: " + " / ".join(f"\"{s}\"" for s in subs) + "."
    return Answer(question, query, kind, text, ev, True)


# --------------------------------------------------------------------------- optional LLM
def llm_available() -> bool:
    if not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CINESCOPE_LLM_MODEL")):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def evidence_prompt(question: str, ans: Answer) -> str:
    lines = []
    for i, e in enumerate(ans.evidence, 1):
        d = e.doc
        lines.append(f"[{i}] {e.span()} (clip {d.get('clip')}, shot {d['shot']}); visual match "
                     f"{e.clip_score:.3f} (median-relative {e.contrast:+.3f}); tags: "
                     f"{', '.join(t['tag'] for t in d.get('tags', []))}"
                     + (f"; subtitles: {' / '.join(s['text'] for s in d['subtitles'])}" if d.get("subtitles") else "")
                     + (f"; characters: {', '.join(c['name'] for c in d['characters'])}" if d.get("characters") else ""))
    return ("You answer questions about a video using ONLY the retrieved evidence below and the attached "
            "key frames (one per evidence item, in order). Cite timestamps like [0:17.3–0:25.4]. If the "
            "evidence does not show the answer, say so plainly. Visual-match scores are CLIP cosine "
            "similarities: low median-relative values mean a weak match.\n\n"
            f"Question: {question}\n\nEvidence:\n" + "\n".join(lines))


def llm_answer(question: str, ans: Answer, frame_paths: list[Path | None], max_images: int = 5) -> Answer:
    """Send the evidence (and key frames) to the model named by CINESCOPE_LLM_MODEL."""
    import anthropic

    content: list[dict] = []
    for pth in frame_paths[:max_images]:
        if pth is not None and Path(pth).exists():
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": base64.standard_b64encode(Path(pth).read_bytes()).decode()}})
    content.append({"type": "text", "text": evidence_prompt(question, ans)})
    client = anthropic.Anthropic()
    resp = client.messages.create(model=os.environ["CINESCOPE_LLM_MODEL"], max_tokens=2000,
                                  messages=[{"role": "user", "content": content}])
    if resp.stop_reason == "refusal":
        return ans
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return Answer(question, ans.query, ans.kind, text or ans.text, ans.evidence, ans.confident, "llm")


def answer(question: str, index, encoder, p: RagParams | None = None, use_llm: bool | None = None) -> Answer:
    p = p or RagParams()
    query, ev = retrieve(question, index, encoder, p)
    ans = template_answer(question, query, ev, p)
    if use_llm is None:
        use_llm = llm_available()
    if use_llm:
        try:
            ans = llm_answer(question, ans, [index.frame_path(e.doc, 1) for e in ev])
        except Exception as exc:  # noqa: BLE001 — any API failure falls back to the template answer
            ans.text += f" (LLM call failed: {type(exc).__name__}; template answer shown.)"
    return ans
