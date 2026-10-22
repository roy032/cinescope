import { fmtTime } from "../util.js";
import Scrub from "./Scrub.jsx";
import { Fact } from "./ShotStats.jsx";

// "How it is built": scenes, the three acts and the beats of the structure (plot
// points, midpoint, climax) over an intensity curve made of the cutting rate, the
// loudness and the motion.
export default function Story({ story, duration, time, onSeek }) {
  const s = story.summary;
  const iv = story.intensity;
  if (!s || !iv?.value?.length) return null;
  const W = 1000;
  const H = 64;
  const x = (t) => (W * t) / Math.max(duration, 1e-3);
  const y = (v) => H - 4 - (H - 10) * v;
  const pts = iv.value.map((v, i) => `${x(iv.t0 + i * iv.dt).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const last = iv.t0 + (iv.value.length - 1) * iv.dt;
  const at = (t) => iv.value[Math.min(iv.value.length - 1, Math.max(0, Math.round((t - iv.t0) / iv.dt)))];
  const climax = story.beats.find((b) => b.label === "climax");
  return (
    <div className="story">
      <div className="facts five">
        <Fact k="scenes" v={s.scenes} title="runs of shots in one place and time" />
        <Fact k="average scene" v={`${s.mean_scene_s.toFixed(0)} s`} title={`longest ${s.longest_scene_s.toFixed(0)} s`} />
        <Fact k="acts" v={s.act_shares.map((a) => `${Math.round(100 * a)}`).join(" / ") + " %"} title="share of the running time in acts 1, 2 and 3" />
        <Fact k="arc" v={s.arc} title="is the last tenth more intense than the first?" />
        <Fact k="climax" v={climax ? fmtTime(climax.t) : "–"} title="the intensity peak in the last 40 %" />
      </div>
      {s.structure === "short" && (
        <p className="dim small">A short clip: the acts describe its shape rather than a plot.</p>
      )}
      <h3>
        Structure <small className="dim">intensity from {s.signals.join(", ")}; shaded acts, ticks at scene changes</small>
      </h3>
      <Scrub
        className="story-lane"
        duration={duration}
        time={time}
        onSeek={onSeek}
        label="story structure"
        describe={(t) => {
          const act = story.acts.find((a) => t >= a.t0 && t <= a.t1);
          const sc = story.scenes.find((e) => t >= e.t0 && t < e.t1);
          return `act ${act ? act.act : "?"} · scene ${sc ? sc.scene + 1 : "?"} · intensity ${at(t).toFixed(2)}`;
        }}
      >
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
          {story.acts.map((a) => (
            <rect key={a.act} x={x(a.t0)} y="0" width={Math.max(x(a.t1) - x(a.t0), 1)} height={H} className={`act act${a.act}`} />
          ))}
          <polygon points={`${x(iv.t0)},${H} ${pts} ${x(last)},${H}`} className="area" />
          {story.scenes.slice(1).map((e) => (
            <line key={e.scene} x1={x(e.t0)} x2={x(e.t0)} y1="0" y2="10" className="scene-tick" />
          ))}
          {story.beats.map((b) => (
            <line key={b.label} x1={x(b.t)} x2={x(b.t)} y1="0" y2={H} className="beat-line" />
          ))}
        </svg>
      </Scrub>
      <div className="legend">
        {story.beats.map((b) => (
          <button key={b.label} className="link small" onClick={() => onSeek(b.t, true)}>
            {b.label} {fmtTime(b.t)}
          </button>
        ))}
      </div>
    </div>
  );
}
