import Scrub from "./Scrub.jsx";
import { Fact } from "./ShotStats.jsx";

// "Where the eye goes": a heat map of the centre of interest over the frame with
// the rule-of-thirds lines, and each shot's placement over time.
const PLACE = { centre: "var(--on)", thirds: "var(--ok)", "power point": "var(--accent)", edge: "var(--talk)" };

export default function Composition({ shots, summary, duration, time, onSeek }) {
  if (!summary?.share) return null;
  const heat = summary.heatmap;
  const n = heat.length;
  const top = Math.max(...heat.flat(), 1e-6);
  const W = 1000;
  const x = (t) => (W * t) / Math.max(duration, 1e-3);
  const pct = (v) => `${Math.round(100 * v)}%`;
  const withComp = shots.filter((s) => s.composition);
  return (
    <div className="composition">
      <p className="desc">{summary.description}</p>
      <div className="comp-row">
        <svg className="heat" viewBox="0 0 160 90" role="img" aria-label="where the centre of interest sits">
          {heat.map((row, r) =>
            row.map((v, c) => (
              <rect key={`${r}-${c}`} x={(160 * c) / n} y={(90 * r) / n} width={160 / n + 0.3} height={90 / n + 0.3}
                style={{ fill: "var(--accent)", opacity: (0.9 * v) / top }} />
            )),
          )}
          {[1, 2].map((k) => (
            <g key={k} className="thirds">
              <line x1={(160 * k) / 3} x2={(160 * k) / 3} y1="0" y2="90" />
              <line x1="0" x2="160" y1={(90 * k) / 3} y2={(90 * k) / 3} />
            </g>
          ))}
        </svg>
        <div className="facts five">
          <Fact k="on the thirds" v={pct(summary.thirds_share)} title="centre of interest near a third line (screen time)" />
          <Fact k="centred" v={pct(summary.share.centre)} />
          <Fact k="near the edge" v={pct(summary.share.edge)} />
          <Fact k="shallow focus" v={pct(summary.shallow_share)} title="sharp subject on a softer background" />
          <Fact k="balance" v={summary.balance > 0.05 ? "right-heavy" : summary.balance < -0.05 ? "left-heavy" : "even"} title={`right minus left saliency: ${summary.balance}`} />
        </div>
      </div>
      <h3>
        Placement per shot <small className="dim">hover to read, click to jump</small>
      </h3>
      <Scrub
        className="lane"
        duration={duration}
        time={time}
        onSeek={onSeek}
        label="composition per shot"
        describe={(t) => {
          const s = withComp.find((it) => t >= it.t0 && t < it.t1);
          return s ? `${s.composition.placement}${s.composition.shallow ? ", shallow focus" : ""}` : null;
        }}
      >
        <svg viewBox={`0 0 ${W} 28`} preserveAspectRatio="none">
          <rect x="0" y="0" width={W} height="28" className="lane-bg" />
          {withComp.map((s) => (
            <rect key={s.index} x={x(s.t0)} y="6" width={Math.max(x(s.t1) - x(s.t0) - 0.5, 1)} height="16"
              style={{ fill: PLACE[s.composition.placement] }} />
          ))}
        </svg>
      </Scrub>
      <div className="legend">
        {Object.entries(PLACE).map(([l, c]) => (
          <span key={l}>
            <span className="sw" style={{ background: c }} /> {l}
          </span>
        ))}
      </div>
    </div>
  );
}
