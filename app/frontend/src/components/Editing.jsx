import { fmtTime } from "../util.js";
import Scrub from "./Scrub.jsx";
import { Fact } from "./ShotStats.jsx";

// "How it was cut": the pacing curve (cuts per minute over time), the editing
// patterns found (shot/reverse-shot exchanges, montages, long takes) and how much
// of the clip cuts back to a camera set-up it has used before.
const COLORS = { "shot/reverse-shot": "var(--on)", montage: "var(--talk)", "long take": "var(--ok)" };

export default function Editing({ editing, duration, time, onSeek }) {
  const { pacing, summary, timeline } = editing;
  if (!pacing?.t?.length) return null;
  const W = 1000;
  const H = 70;
  const maxC = Math.max(...pacing.cpm, 1);
  const x = (t) => (W * t) / Math.max(duration, 1e-3);
  const y = (c) => H - 2 - ((H - 6) * c) / maxC;
  const line = pacing.t.map((t, i) => `${x(t).toFixed(1)},${y(pacing.cpm[i]).toFixed(1)}`).join(" ");
  const at = (t) => {
    const i = Math.min(pacing.t.length - 1, Math.max(0, Math.round((t / Math.max(duration, 1e-3)) * (pacing.t.length - 1))));
    return pacing.cpm[i];
  };
  const pattern = (t) => timeline.find((e) => t >= e.t0 && t < e.t1);
  return (
    <div className="editing">
      <div className="facts five">
        <Fact k="camera set-ups" v={summary.setups} title="distinct camera angles the editor cuts between" />
        <Fact k="set-ups reused" v={summary.recurring_setups} title={`${summary.shots_in_recurring_setups} shots return to an angle used before`} />
        <Fact k="shot/reverse-shot" v={summary.exchanges} />
        <Fact k="montages" v={summary.montages} />
        {summary.fastest_t != null ? (
          <button className="fact as-button" onClick={() => onSeek(Math.max(0, summary.fastest_t - pacing.window_s / 2), true)} title="play the fastest-cut stretch">
            <div className="v">{summary.fastest_cpm.toFixed(0)} cuts/min</div>
            <div className="k">fastest, at {fmtTime(summary.fastest_t)} ▶</div>
          </button>
        ) : (
          <Fact k="long takes" v={summary.long_takes} />
        )}
      </div>
      <h3>
        Pacing <small className="dim">cuts per minute over a {pacing.window_s.toFixed(0)} s window</small>
      </h3>
      <Scrub
        className="pacing"
        duration={duration}
        time={time}
        onSeek={onSeek}
        label="pacing"
        describe={(t) => {
          const p = pattern(t);
          return `${at(t).toFixed(0)} cuts/min${p ? ` · ${p.label}` : ""}`;
        }}
      >
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
          {timeline.map((e, i) => (
            <rect key={i} x={x(e.t0)} y="0" width={Math.max(x(e.t1) - x(e.t0), 2)} height={H} style={{ fill: COLORS[e.label] }} className="band" />
          ))}
          <polygon points={`0,${H} ${line} ${W},${H}`} className="area" />
          <polyline points={line} className="curve" />
        </svg>
      </Scrub>
      <div className="legend">
        {Object.entries(COLORS).map(([l, c]) => (
          <span key={l}>
            <span className="sw" style={{ background: c }} /> {l} ({timeline.filter((e) => e.label === l).length})
          </span>
        ))}
      </div>
    </div>
  );
}
