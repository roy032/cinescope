import Scrub from "./Scrub.jsx";
import { Fact } from "./ShotStats.jsx";

// "How the cuts are put together": how many camera set-ups the editor cuts between
// and how often a shot returns to one, and the patterns the cuts make over time
// (shot/reverse-shot exchanges, montages, long takes). The cutting rate itself is
// the pace curve in the Shots panel.
const COLORS = { "shot/reverse-shot": "var(--on)", montage: "var(--talk)", "long take": "var(--ok)" };

export default function Editing({ editing, duration, time, onSeek }) {
  const { summary, timeline } = editing;
  if (!summary) return null;
  const W = 1000;
  const x = (t) => (W * t) / Math.max(duration, 1e-3);
  return (
    <div className="editing">
      <div className="facts five">
        <Fact k="camera set-ups" v={summary.setups} title="distinct camera angles the editor cuts between" />
        <Fact k="set-ups reused" v={summary.recurring_setups} title={`${summary.shots_in_recurring_setups} shots return to an angle used before`} />
        <Fact k="shot/reverse-shot" v={summary.exchanges} title="runs of four or more shots alternating between two angles" />
        <Fact k="montages" v={summary.montages} title="runs of five or more shots much shorter than this clip's median shot" />
        <Fact k="long takes" v={summary.long_takes} title="shots at least four times the median and 15 s or longer" />
      </div>
      <h3>
        Patterns <small className="dim">hover to read, click to jump</small>
      </h3>
      <Scrub
        className="lane"
        duration={duration}
        time={time}
        onSeek={onSeek}
        label="editing patterns"
        describe={(t) => {
          const e = timeline.find((it) => t >= it.t0 && t < it.t1);
          return e ? `${e.label}, ${e.shots} shot${e.shots > 1 ? "s" : ""}` : null;
        }}
      >
        <svg viewBox={`0 0 ${W} 28`} preserveAspectRatio="none">
          <rect x="0" y="0" width={W} height="28" className="lane-bg" />
          {timeline.map((e, i) => (
            <rect key={i} x={x(e.t0)} y="6" width={Math.max(x(e.t1) - x(e.t0), 1.5)} height="16" rx="2" style={{ fill: COLORS[e.label] }} />
          ))}
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
