import { useState } from "react";
import { api } from "../api.js";
import Scrub from "./Scrub.jsx";

const inside = (spans, t) => spans.some(([a, b]) => t >= a && t <= b);

// One row per character: face, on-screen spans (bars) and talking spans (marks on top).
export default function CharacterTimeline({ clip, chars, duration, time, onSeek, maxRows = 10 }) {
  const [all, setAll] = useState(false);
  if (!chars.length) return <p className="dim">No faces were tracked long enough to count as characters.</p>;
  const rows = all ? chars : chars.slice(0, maxRows);
  const W = 1000;
  const x = (t) => (W * t) / Math.max(duration, 1e-3);
  return (
    <div className="chars">
      {rows.map((c) => {
        const now = inside(c.on_screen, time);
        return (
          <div className={`char-row ${now ? "now" : ""}`} key={c.id}>
            <button
              className="who"
              onClick={() => c.on_screen.length && onSeek(c.on_screen[0][0] + 0.01)}
              title={`${c.id}: jump to the first appearance`}
            >
              <img src={api.file(clip, c.thumb)} alt="" loading="lazy" />
              <span>
                <b>{c.name || c.id}</b>
                <span className="dim small">
                  {" "}
                  {c.screen_time_s.toFixed(1)} s on screen
                  <br />
                  {c.speaking_time_s > 0 ? `${c.speaking_time_s.toFixed(1)} s talking` : "not heard talking"}
                </span>
              </span>
            </button>
            <Scrub
              className="lane"
              duration={duration}
              time={time}
              onSeek={onSeek}
              label={`${c.id} timeline`}
              describe={(t) => (inside(c.speaking, t) ? "talking" : inside(c.on_screen, t) ? "on screen" : null)}
            >
              <svg viewBox={`0 0 ${W} 28`} preserveAspectRatio="none">
                <rect x="0" y="0" width={W} height="28" className="lane-bg" />
                {c.on_screen.map(([a, b], i) => (
                  <rect key={`o${i}`} x={x(a)} y="6" width={Math.max(x(b) - x(a), 1.5)} height="16" rx="2" className="on" />
                ))}
                {c.speaking.map(([a, b], i) => (
                  <rect key={`s${i}`} x={x(a)} y="10" width={Math.max(x(b) - x(a), 1.5)} height="8" rx="2" className="talk" />
                ))}
              </svg>
            </Scrub>
          </div>
        );
      })}
      <div className="legend">
        <span className="sw on" /> on screen <span className="sw talk" /> talking
        {chars.length > maxRows && (
          <button className="link" onClick={() => setAll(!all)}>
            {all ? "show fewer" : `show all ${chars.length} characters`}
          </button>
        )}
        <span className="dim"> · click a face to jump to its first appearance</span>
      </div>
    </div>
  );
}
