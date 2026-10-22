import { useEffect, useState } from "react";
import { api } from "../api.js";
import { fmtTime, reelStep } from "../util.js";
import Scrub from "./Scrub.jsx";
import { Fact } from "./ShotStats.jsx";

// The strongest moments of the clip as a short reel: play it here (the player
// jumps from moment to moment), or download it rendered as an MP4 or as an EDL
// to refine in an editor.
export default function Highlights({ clip, highlights, shots, duration, time, onSeek, onPause }) {
  const [playing, setPlaying] = useState(null); // index of the moment being played
  const moments = highlights.moments || [];
  const s = highlights.summary || {};

  useEffect(() => {
    if (playing == null) return;
    const step = reelStep(moments, playing, time);
    if (step.action === "jump") {
      setPlaying(step.index);
      onSeek(step.t, true);
    } else if (step.action === "stop") {
      setPlaying(null);
      onPause?.();
    }
  }, [time]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => setPlaying(null), [clip]);

  if (!moments.length) return <p className="dim">No highlights were found in this clip.</p>;
  const W = 1000;
  const x = (t) => (W * t) / Math.max(duration, 1e-3);
  const play = () => {
    setPlaying(0);
    onSeek(moments[0].t0, true);
  };
  const stop = () => {
    setPlaying(null);
    onPause?.();
  };
  const thumb = (m) => shots?.[m.shots?.[0]]?.thumb;
  return (
    <div className="highlights">
      <div className="facts five">
        <Fact k="moments" v={moments.length} />
        <Fact k="reel" v={fmtTime(s.duration_s || 0)} />
        <Fact k="of the clip" v={`${Math.round(100 * (s.share || 0))}%`} />
        <Fact k="scored by" v={(s.signals || []).join(", ") || "–"} />
        <Fact k="edges on the beat" v={s.on_beat ? "yes" : "no beat"} />
      </div>
      <div className="hl-actions">
        {playing == null ? (
          <button className="primary" onClick={play} title="play the moments one after another in the player">
            ▶ Play highlights
          </button>
        ) : (
          <button className="primary" onClick={stop}>
            ■ Stop ({playing + 1}/{moments.length})
          </button>
        )}
        <a className="btn-ghost" href={api.exportUrl(clip, "mp4")} download title="rendered with sound; the first download takes a moment">
          Download MP4
        </a>
        <a className="btn-ghost" href={api.exportUrl(clip, "highlights")} download title="CMX 3600, to rebuild the reel in an editor">
          EDL
        </a>
      </div>
      <Scrub
        className="lane"
        duration={duration}
        time={time}
        onSeek={onSeek}
        label="highlights"
        describe={(t) => {
          const m = moments.find((e) => t >= e.t0 && t <= e.t1);
          return m ? (m.climax ? "climax" : "highlight") : null;
        }}
      >
        <svg viewBox={`0 0 ${W} 28`} preserveAspectRatio="none">
          <rect x="0" y="0" width={W} height="28" className="lane-bg" />
          {moments.map((m, i) => (
            <rect
              key={i}
              x={x(m.t0)}
              y="0"
              width={Math.max(x(m.t1) - x(m.t0), 2)}
              height="28"
              className={`hl-moment${m.climax ? " climax" : ""}${i === playing ? " now" : ""}`}
            />
          ))}
        </svg>
      </Scrub>
      <ol className="hl-list">
        {moments.map((m, i) => (
          <li key={i} className={i === playing ? "now" : ""}>
            <button className="link" onClick={() => onSeek(m.t0, true)} title="play from here">
              {thumb(m) && <img src={api.file(clip, thumb(m))} alt="" loading="lazy" />}
              <span className="tnum">
                {fmtTime(m.t0)} – {fmtTime(m.t1)}
              </span>
              <span className="dim small">
                {m.climax ? "climax · " : ""}
                {(m.t1 - m.t0).toFixed(1)} s · score {m.score.toFixed(2)}
              </span>
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}
