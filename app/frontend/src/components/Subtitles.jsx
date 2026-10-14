import { useEffect, useRef } from "react";
import { fmtTime } from "../api.js";

// Subtitle cues read from the picture (current one highlighted), then other on-screen text.
export default function Subtitles({ subs, time, onSeek }) {
  const box = useRef(null);
  const cur = subs.cues.findIndex((c) => time >= c.t0 && time < c.t1);

  useEffect(() => {
    const el = box.current?.querySelector(".cue.current");
    if (el && box.current) box.current.scrollTop = el.offsetTop - box.current.clientHeight / 2;
  }, [cur]);

  return (
    <div className="subs" ref={box}>
      {!subs.cues.length && <p className="dim">No burned-in subtitles found.</p>}
      {subs.cues.map((c, i) => (
        <button key={i} className={`cue ${i === cur ? "current" : ""}`} onClick={() => onSeek(c.t0 + 0.01)}>
          <span className="t">{fmtTime(c.t0)}</span>
          <span className="text">{c.text}</span>
        </button>
      ))}
      {subs.credits?.length > 0 && (
        <>
          <h3>Other on-screen text</h3>
          {subs.credits.map((c, i) => (
            <button key={`c${i}`} className="cue credit" onClick={() => onSeek(c.t + 0.01)}>
              <span className="t">{fmtTime(c.t)}</span>
              <span className="text">{c.text}</span>
            </button>
          ))}
        </>
      )}
    </div>
  );
}
