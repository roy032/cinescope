import { useState } from "react";
import { api, fmtTime } from "../api.js";

// Scene search in plain English (Phase 8), and "ask the movie" (retrieval + a template answer).
export default function SearchPanel({ clip, report, onSeek }) {
  const [q, setQ] = useState("");
  const [mode, setMode] = useState("search");
  const [res, setRes] = useState(null);
  const [answer, setAnswer] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  if (!report.search) {
    return (
      <div className="panel search">
        <h2>Scene search</h2>
        <p className="dim">Run the search stage to search this clip in plain English.</p>
      </div>
    );
  }

  const go = async (e) => {
    e.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    setErr("");
    try {
      if (mode === "search") {
        setRes(await api.search(clip, q.trim()));
        setAnswer(null);
      } else {
        setAnswer(await api.ask(clip, q.trim()));
        setRes(null);
      }
    } catch (ex) {
      setErr(String(ex));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel search">
      <h2>
        Scene search <small>{report.search.n_docs} segments indexed</small>
      </h2>
      <form onSubmit={go}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder={mode === "search" ? "e.g. a man walks away from an explosion" : "e.g. when does the plane take off?"}
          aria-label="search query"
        />
        <select value={mode} onChange={(e) => setMode(e.target.value)} aria-label="mode">
          <option value="search">search</option>
          <option value="ask">ask</option>
        </select>
        <button className="primary" disabled={busy}>
          {busy ? "…" : "Go"}
        </button>
      </form>
      {err && <div className="error small">{err}</div>}
      {answer && (
        <div className="answer">
          <p>{answer.answer}</p>
        </div>
      )}
      {res && (
        <div className="hits">
          {res.hits.map((h) => (
            <button key={`${h.shot}-${h.part}`} className="hit" onClick={() => onSeek(h.t0 + 0.01)}>
              <img src={h.frames[Math.floor(h.frames.length / 2)] || h.frames[0]} alt="" />
              <div className="hit-meta">
                <span>
                  #{h.rank} · {fmtTime(h.t0)}
                </span>
                <span className="dim">{h.score.toFixed(3)}</span>
              </div>
              <div className="dim small tags">{h.tags.slice(0, 2).join(", ")}</div>
            </button>
          ))}
          <div className="dim small">{res.ms} ms</div>
        </div>
      )}
    </div>
  );
}
