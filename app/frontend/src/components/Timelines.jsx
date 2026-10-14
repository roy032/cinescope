import { fmtTime } from "../api.js";

// Timelines of stages the app does not know by name (report.timelines: Phase 4's
// cutouts, Phase 6's scene types ...): one lane per stage, one labelled span per item.
const hue = (s) => [...String(s)].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 7);

export default function Timelines({ timelines, duration, time, onSeek }) {
  const W = 1000;
  const x = (t) => (W * t) / Math.max(duration, 1e-3);
  return (
    <div className="chars">
      {Object.entries(timelines).map(([name, items]) => (
        <div className="char-row" key={name}>
          <div className="who">
            <b>{name}</b>
          </div>
          <svg viewBox={`0 0 ${W} 28`} preserveAspectRatio="none" className="lane">
            <rect x="0" y="0" width={W} height="28" className="lane-bg" />
            {items.map((it, i) => (
              <rect key={i} x={x(it.t0)} y="6" width={Math.max(x(it.t1) - x(it.t0), 1.5)} height="16"
                style={{ fill: `hsl(${hue(it.label)} 55% 55%)` }} onClick={() => onSeek(it.t0 + 0.01)}>
                <title>{`${it.label ?? ""} ${fmtTime(it.t0)}–${fmtTime(it.t1)}`}</title>
              </rect>
            ))}
            <line x1={x(time)} x2={x(time)} y1="0" y2="28" className="ph" />
          </svg>
          <div />
          <div className="legend">
            {[...new Set(items.map((it) => it.label))].map((l) => (
              <span key={l}>
                <span className="sw" style={{ background: `hsl(${hue(l)} 55% 55%)` }} /> {l}
              </span>
            ))}
            {!items.length && <span>nothing detected</span>}
          </div>
        </div>
      ))}
    </div>
  );
}
