import { useMemo } from "react";
import { lumaColor, shotAt, warmthColor } from "../util.js";
import Scrub from "./Scrub.jsx";
import { Fact } from "./ShotStats.jsx";

// "How it was lit and graded": the clip's overall look in a colourist's words, and
// three lanes over time, one cell per shot: exposure (the shot's mean brightness as
// grey), colour temperature (blue cool .. amber warm) and saturation (bar height).
const pct = (v) => `${Math.round(100 * (Number(v) || 0))}%`;

export default function Look({ shots, summary, duration, time, onSeek }) {
  const rows = useMemo(() => shots.filter((s) => s.look), [shots]);
  if (!summary || !rows.length) return null;
  const W = 1000;
  const x = (t) => (W * t) / Math.max(duration, 1e-3);
  const w = (s) => Math.max(x(s.t1) - x(s.t0) - 0.5, 0.8);
  const maxSat = Math.max(...rows.map((s) => s.look.saturation), 0.05);
  const describe = (t) => {
    const i = shotAt(shots, t);
    const lk = i >= 0 ? shots[i].look : null;
    return lk
      ? `shot #${shots[i].index}: ${lk.key}, ${lk.temperature}, brightness ${pct(lk.brightness)}, saturation ${pct(lk.saturation)}`
      : null;
  };
  const lane = (name, children, h = 22) => (
    <div className="look-lane">
      <span className="dim small">{name}</span>
      <Scrub className="lane" duration={duration} time={time} onSeek={onSeek} label={`${name} over time`} describe={describe}>
        <svg viewBox={`0 0 ${W} ${h}`} preserveAspectRatio="none" style={{ height: h }}>
          <rect x="0" y="0" width={W} height={h} className="lane-bg" />
          {children(h)}
        </svg>
      </Scrub>
    </div>
  );
  return (
    <div className="look">
      <p className="look-desc">{summary.description}</p>
      <div className="facts five">
        <Fact k="lighting key" v={summary.key} title={share(summary.key_share)} />
        <Fact k="colour temperature" v={`${summary.temperature} (${summary.warmth.toFixed(1)})`} title={share(summary.temperature_share)} />
        <Fact k="contrast" v={pct(summary.contrast * 2)} title="RMS contrast of luma, doubled so a hard black-and-white frame reads 100%" />
        <Fact k="saturation" v={pct(summary.saturation)} />
        <Fact k="teal & orange" v={pct(summary.teal_orange)} title="how much of the colour sits in the two blockbuster-grade bands" />
      </div>
      {lane("exposure", (h) => rows.map((s) => <rect key={s.index} x={x(s.t0)} y="0" width={w(s)} height={h} style={{ fill: lumaColor(s.look.brightness) }} />))}
      {lane("temperature", (h) => rows.map((s) => <rect key={s.index} x={x(s.t0)} y="0" width={w(s)} height={h} style={{ fill: warmthColor(s.look.warmth) }} />))}
      {lane(
        "saturation",
        (h) =>
          rows.map((s) => {
            const bh = Math.max(1, (h - 2) * (s.look.saturation / maxSat));
            return <rect key={s.index} x={x(s.t0)} y={h - bh} width={w(s)} height={bh} className="sat" />;
          }),
        30,
      )}
      <div className="legend">
        symmetry {pct(summary.symmetry)} · tonal range {pct(summary.range)} · crushed blacks {pct(summary.crushed)} · blown
        highlights {pct(summary.clipped)}
      </div>
    </div>
  );
}

function share(s) {
  return s ? Object.entries(s).map(([k, v]) => `${k} ${Math.round(100 * v)}%`).join(", ") : undefined;
}
