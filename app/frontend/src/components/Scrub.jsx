import { useState } from "react";
import { clamp, fmtTime } from "../util.js";

// A strip that maps x to time: hover shows the time (and `describe(t)` if given)
// under a hairline, click seeks, the playhead marks `time`, and the arrow keys
// step through it when focused. Barcode, character lanes and timelines share it.
export default function Scrub({ duration, time, onSeek, describe, label, className = "", children }) {
  const [hover, setHover] = useState(null); // {x: 0..1, t}
  const at = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = clamp((e.clientX - r.left) / r.width, 0, 1);
    return { x, t: x * duration };
  };
  const key = (e) => {
    const step = { ArrowLeft: -5, ArrowRight: 5, Home: -Infinity, End: Infinity }[e.key];
    if (step === undefined) return;
    e.preventDefault();
    e.stopPropagation();
    onSeek(clamp(time + step, 0, Math.max(duration - 0.05, 0)));
  };
  const pct = (v) => `${100 * clamp(v, 0, 1)}%`;
  const extra = hover && describe ? describe(hover.t) : null;
  return (
    <div
      className={`scrub ${className}`}
      role="slider"
      tabIndex={0}
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={Math.round(duration)}
      aria-valuenow={Math.round(time)}
      aria-valuetext={fmtTime(time)}
      onMouseMove={(e) => setHover(at(e))}
      onMouseLeave={() => setHover(null)}
      onClick={(e) => onSeek(at(e).t)}
      onKeyDown={key}
    >
      {children}
      {duration > 0 && <div className="playhead" style={{ left: pct(time / duration) }} />}
      {hover && (
        <>
          <div className="hoverline" style={{ left: pct(hover.x) }} />
          <div className={`tip ${hover.x > 0.75 ? "left" : ""}`} style={{ left: pct(hover.x) }}>
            <b>{fmtTime(hover.t)}</b>
            {extra && <span> · {extra}</span>}
          </div>
        </>
      )}
    </div>
  );
}
