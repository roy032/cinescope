import { fmtTime } from "../util.js";
import Scrub from "./Scrub.jsx";
import { Fact } from "./ShotStats.jsx";

// "How it sounds": the mix's loudness in broadcast units (LUFS, EBU R 128), its
// dynamics, how much is speech, whether the picture cuts on the sound, and the
// loudness over time with the peaks and silences marked.
const LO = -60; // the floor of the chart, LUFS

export default function Sound({ sound, duration, time, onSeek }) {
  const s = sound.summary;
  const env = sound.envelope;
  if (!sound.audio || !s || !env?.lufs?.length) return <p className="dim">This clip has no soundtrack.</p>;
  const W = 1000;
  const H = 64;
  const hi = Math.max(-10, ...env.lufs);
  const x = (t) => (W * t) / Math.max(duration, 1e-3);
  const y = (l) => H - ((H - 2) * (Math.max(l, LO) - LO)) / (hi - LO);
  const pts = env.lufs.map((l, i) => `${x(env.t0 + i * env.dt).toFixed(1)},${y(l).toFixed(1)}`).join(" ");
  const at = (t) => env.lufs[Math.min(env.lufs.length - 1, Math.max(0, Math.round((t - env.t0) / env.dt)))];
  const sync = s.cut_sync;
  return (
    <div className="sound">
      <div className="facts five">
        <Fact k="integrated loudness" v={s.integrated_lufs != null ? `${s.integrated_lufs} LUFS` : "silent"} title="EBU R 128, from 16 kHz mono: close to, not certified" />
        <Fact k="loudness range" v={`${s.lra_lu} LU`} title="the spread between quiet and loud passages (LRA)" />
        <Fact k="sample peak" v={`${s.peak_dbfs} dBFS`} />
        <Fact k="speech" v={s.speech_share != null ? `${Math.round(100 * s.speech_share)}%` : "–"} title="share of the running time with a voice-activity score above 0.5" />
        <Fact
          k="cuts on a sound"
          v={sync.cuts ? `${Math.round(100 * sync.on_sound)}%` : "–"}
          title={`cuts within 0.1 s of a sound onset; chance alone would give ${Math.round(100 * sync.chance)}%`}
        />
      </div>
      <h3>
        Loudness <small className="dim">momentary, LUFS; peaks and silences shaded</small>
        <button className="link small spaced" onClick={() => onSeek(Math.max(0, s.loudest_t - 1), true)}>
          play the loudest moment ({fmtTime(s.loudest_t)})
        </button>
      </h3>
      <Scrub
        className="loudness"
        duration={duration}
        time={time}
        onSeek={onSeek}
        label="loudness"
        describe={(t) => {
          const e = sound.timeline.find((it) => t >= it.t0 && t < it.t1);
          return `${at(t).toFixed(0)} LUFS${e ? ` · ${e.label}` : ""}`;
        }}
      >
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
          {sound.timeline.map((e, i) => (
            <rect key={i} x={x(e.t0)} y="0" width={Math.max(x(e.t1) - x(e.t0), 2)} height={H} className={e.label} />
          ))}
          <polygon points={`${x(env.t0)},${H} ${pts} ${x(env.t0 + (env.lufs.length - 1) * env.dt)},${H}`} className="area" />
        </svg>
      </Scrub>
      <div className="legend">
        <span>
          <span className="sw" style={{ background: "var(--talk)" }} /> peaks ({s.peaks})
        </span>
        <span>
          <span className="sw" style={{ background: "var(--on)" }} /> silences ({s.silences}, {s.silence_s} s)
        </span>
        <span>{s.onsets_per_min} sound onsets a minute</span>
      </div>
    </div>
  );
}
