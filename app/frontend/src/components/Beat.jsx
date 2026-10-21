import Scrub from "./Scrub.jsx";
import { Fact } from "./ShotStats.jsx";

// "Is it cut to the music": the tempo, the sections where the soundtrack has a
// clear beat, the beats themselves, and how often a cut lands on one.
export default function Beat({ rhythm, duration, time, onSeek }) {
  const s = rhythm.summary;
  if (!rhythm.audio || !s) return <p className="dim">This clip has no soundtrack.</p>;
  if (!rhythm.timeline.length) return <p className="dim">No section of this soundtrack has a steady beat.</p>;
  const W = 1000;
  const x = (t) => (W * t) / Math.max(duration, 1e-3);
  const cb = s.cuts_on_beat || {};
  return (
    <div className="beat">
      <div className="facts five">
        <Fact k="tempo" v={s.bpm ? `${s.bpm.toFixed(0)} BPM` : "–"} />
        <Fact k="pulse clarity" v={s.pulse_clarity.toFixed(2)} title="how much of the onsets the beat explains, 0..1" />
        <Fact k="music with a beat" v={`${Math.round(100 * s.rhythmic_share)}%`} />
        <Fact k="beats" v={s.beats} />
        <Fact
          k="cuts on the beat"
          v={cb.on_beat != null ? `${Math.round(100 * cb.on_beat)}%` : "–"}
          title={cb.chance != null ? `of ${cb.cuts} cuts in music; chance alone would give ${Math.round(100 * cb.chance)}%` : "no cuts in music"}
        />
      </div>
      <Scrub
        className="lane"
        duration={duration}
        time={time}
        onSeek={onSeek}
        label="beat"
        describe={(t) => (rhythm.timeline.some((e) => t >= e.t0 && t <= e.t1) ? "music with a beat" : null)}
      >
        <svg viewBox={`0 0 ${W} 28`} preserveAspectRatio="none">
          <rect x="0" y="0" width={W} height="28" className="lane-bg" />
          {rhythm.timeline.map((e, i) => (
            <rect key={i} x={x(e.t0)} y="0" width={Math.max(x(e.t1) - x(e.t0), 1)} height="28" className="beat-section" />
          ))}
          {rhythm.beats.map((b, i) => (
            <line key={i} x1={x(b)} x2={x(b)} y1="7" y2="21" className="beat-tick" />
          ))}
        </svg>
      </Scrub>
    </div>
  );
}
