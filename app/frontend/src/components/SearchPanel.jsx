import { forwardRef, useRef, useState } from "react";
import { api, fmtTime } from "../api.js";

const EXAMPLES = {
  search: ["a person talking on the phone", "a car driving at night", "an explosion", "a crowd outdoors", "a close-up of a face"],
  ask: ["when does someone run?", "who talks the most?", "where is the car?"],
};

// Scene search in plain English (Phase 8), and "ask the movie" (retrieval + a template answer).
const SearchPanel = forwardRef(function SearchPanel({ clip, report, onSeek }, inputRef) {
  const [q, setQ] = useState("");
  const [mode, setMode] = useState("search");
  const [res, setRes] = useState(null); // {mode, query, hits?, answer?, ms?}
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const abort = useRef(null);

  if (!report.search) {
    return (
      <div className="panel search">
        <h2>Scene search</h2>
        <p className="dim">
          Run the <b>search</b> stage (Stages ▾ in the top bar) to search this clip in plain English and ask it questions.
        </p>
      </div>
    );
  }

  const run = async (text, m = mode) => {
    const query = text.trim();
    if (!query) return;
    abort.current?.abort();
    const ac = new AbortController();
    abort.current = ac;
    setBusy(true);
    setErr("");
    try {
      if (m === "search") {
        const r = await api.search(clip, query, 8, ac.signal);
        setRes({ mode: m, query, hits: r.hits, ms: r.ms });
      } else {
        const a = await api.ask(clip, query, ac.signal);
        setRes({
          mode: m,
          query,
          answer: a,
          hits: (a.evidence || []).map((e, i) => ({ ...e, rank: i + 1, score: e.clip_score, frames: [] })),
        });
      }
    } catch (ex) {
      if (ex.name !== "AbortError") setErr(ex.message);
    } finally {
      if (abort.current === ac) setBusy(false);
    }
  };

  return (
    <div className="panel search">
      <h2>
        Scene search <small>{report.search.n_docs} segments indexed · press / to type</small>
      </h2>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          run(q);
        }}
      >
        <div className="seg" role="tablist" aria-label="mode">
          {["search", "ask"].map((m) => (
            <button
              key={m}
              type="button"
              role="tab"
              aria-selected={mode === m}
              className={mode === m ? "on" : ""}
              onClick={() => setMode(m)}
            >
              {m === "search" ? "Find scenes" : "Ask"}
            </button>
          ))}
        </div>
        <div className="qbox">
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={mode === "search" ? "describe a moment, e.g. a man walks away from an explosion" : "ask about the clip, e.g. when does the plane take off?"}
            aria-label="search query"
            type="search"
          />
          <button className="primary" disabled={busy || !q.trim()}>
            {busy ? <span className="spinner small" aria-label="searching" /> : "Go"}
          </button>
        </div>
      </form>
      {!res && !busy && (
        <div className="examples">
          <span className="dim small">Try:</span>
          {EXAMPLES[mode].map((ex) => (
            <button
              key={ex}
              className="chip-btn"
              onClick={() => {
                setQ(ex);
                run(ex);
              }}
            >
              {ex}
            </button>
          ))}
        </div>
      )}
      {err && <div className="error small">{err}</div>}
      {res?.answer && (
        <div className={`answer ${res.answer.confident ? "" : "unsure"}`}>
          <p>{res.answer.answer}</p>
          {!res.answer.confident && <p className="dim small">Low confidence: check the scenes below.</p>}
        </div>
      )}
      {res?.hits && (
        <>
          {!res.hits.length && <p className="dim">Nothing matched “{res.query}”.</p>}
          <div className="hits">
            {res.hits.map((h) => {
              const img = h.frames?.length ? h.frames[Math.floor(h.frames.length / 2)] : api.frame(clip, (h.t0 + h.t1) / 2, 320);
              const detail = [
                h.characters?.length ? `with ${h.characters.join(", ")}` : "",
                h.subtitles?.length ? `“${h.subtitles.join(" ")}”` : "",
              ]
                .filter(Boolean)
                .join(" · ");
              return (
                <button key={`${h.shot}-${h.part}-${h.rank}`} className="hit" onClick={() => onSeek(h.t0 + 0.01, true)} title={detail || `shot ${h.shot}`}>
                  <img src={img} alt="" loading="lazy" />
                  <div className="hit-meta">
                    <span>
                      <b>{h.rank}</b> · {fmtTime(h.t0)}
                    </span>
                    <span className="dim" title="similarity">
                      {h.score?.toFixed?.(2)}
                    </span>
                  </div>
                  {h.tags?.length > 0 && <div className="dim small tags">{h.tags.slice(0, 2).join(", ")}</div>}
                </button>
              );
            })}
          </div>
          {res.ms != null && <div className="dim small">{res.hits.length} results in {res.ms} ms</div>}
        </>
      )}
    </div>
  );
});

export default SearchPanel;
