import { useMemo } from "react";
import { fmtTime } from "../api.js";
import { pacePath, shotAt, shotStats } from "../util.js";
import Scrub from "./Scrub.jsx";

// "How it was shot": the editor's numbers (average shot length, cuts per minute),
// the cutting rhythm (one bar per shot, as long as the shot lasts and as tall), and
// the pace (cuts per minute over time, from report.pacing), and how screen time
// splits across framings, camera moves and scene types.
export default function ShotStats({ shots, pacing, duration, time, onSeek }) {
  const st = useMemo(() => shotStats(shots, duration), [shots, duration]);
  if (!st) return null;
  const cur = shotAt(shots, time);
  const maxD = Math.max(...shots.map((s) => s.duration_s), 1e-3);
  const W = 1000;
  const H = 60;
  const x = (t) => (W * t) / Math.max(duration, 1e-3);
  return (
    <div className="shotstats">
      <div className="facts five">
        <Fact v={st.n} k="shots" />
        <Fact v={`${st.asl.toFixed(1)} s`} k="average shot" />
        <Fact v={`${st.median.toFixed(1)} s`} k="median shot" />
        <Fact v={st.cutsPerMin.toFixed(1)} k="cuts per minute" />
        <button className="fact as-button" onClick={() => onSeek(st.longest.t0 + 0.01, true)} title="play the longest shot">
          <div className="v">{st.longest.duration_s.toFixed(1)} s</div>
          <div className="k">longest (#{st.longest.index}, {fmtTime(st.longest.t0)}) ▶</div>
        </button>
      </div>
      <h3>Cutting rhythm <small className="dim">each bar is a shot: its width and height are its length</small></h3>
      <Scrub
        className="rhythm"
        duration={duration}
        time={time}
        onSeek={onSeek}
        label="cutting rhythm"
        describe={(t) => {
          const i = shotAt(shots, t);
          return i < 0 ? null : `shot #${shots[i].index}, ${shots[i].duration_s.toFixed(1)} s`;
        }}
      >
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
          {shots.map((s, i) => {
            const h = Math.max(2, (H - 2) * Math.sqrt(s.duration_s / maxD));
            return (
              <rect
                key={s.index}
                x={x(s.t0) + 0.5}
                y={H - h}
                width={Math.max(x(s.t1) - x(s.t0) - 1, 0.8)}
                height={h}
                rx="1.5"
                className={i === cur ? "cur" : ""}
              />
            );
          })}
        </svg>
      </Scrub>
      {pacing?.curve && (
        <>
          <h3>
            Pace{" "}
            <small className="dim">
              cuts per minute over a {pacing.window_s.toFixed(0)} s window · fastest {pacing.fastest.cuts_per_min.toFixed(0)} at{" "}
              <button className="link" onClick={() => onSeek(pacing.fastest.t, true)}>
                {fmtTime(pacing.fastest.t)}
              </button>
            </small>
          </h3>
          <Scrub
            className="pace"
            duration={duration}
            time={time}
            onSeek={onSeek}
            label="pace"
            describe={(t) => {
              const c = pacing.curve;
              let i = 0;
              while (i + 1 < c.t.length && c.t[i + 1] <= t) i += 1;
              return `${c.cuts_per_min[i].toFixed(0)} cuts per minute`;
            }}
          >
            <svg viewBox={`0 0 ${W} 42`} preserveAspectRatio="none">
              <path className="fill" d={pacePath(pacing.curve, duration, W, 42, true)} />
              <path className="line" d={pacePath(pacing.curve, duration, W, 42)} />
            </svg>
          </Scrub>
        </>
      )}
      <div className="dists">
        {st.framing.length > 0 && <Dist title="Framing" rows={st.framing} />}
        {st.camera.length > 0 && <Dist title="Camera" rows={st.camera} />}
        {st.scenes.length > 0 && <Dist title="Scene type" rows={st.scenes} />}
      </div>
    </div>
  );
}

function Dist({ title, rows }) {
  return (
    <div className="dist">
      <h3>
        {title} <small className="dim">share of screen time</small>
      </h3>
      {rows.slice(0, 6).map((r) => (
        <div className="drow" key={r.label} title={`${r.label}: ${r.secs.toFixed(1)} s`}>
          <span className="dl">{r.label}</span>
          <span className="db">
            <span style={{ width: `${Math.max(1, 100 * r.frac)}%` }} />
          </span>
          <span className="dv">{Math.round(100 * r.frac)}%</span>
        </div>
      ))}
    </div>
  );
}

export function Fact({ k, v, title }) {
  return (
    <div className="fact" title={title}>
      <div className="v">{v}</div>
      <div className="k">{k}</div>
    </div>
  );
}
