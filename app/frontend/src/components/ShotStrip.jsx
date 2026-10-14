import { useEffect, useRef } from "react";
import { api, fmtTime } from "../api.js";

const FRAMING_SHORT = { ECS: "ECU", CS: "CU", MS: "MS", FS: "FS", LS: "LS" };

// One card per shot: thumbnail, time, framing and camera-move labels, palette.
export default function ShotStrip({ clip, shots, time, onSeek }) {
  const cur = shots.findIndex((s) => time >= s.t0 && time < s.t1);
  const strip = useRef(null);

  useEffect(() => {
    const el = strip.current?.querySelector(`[data-shot="${cur}"]`);
    if (el && strip.current) {
      const s = strip.current;
      const left = el.offsetLeft - s.clientWidth / 2 + el.clientWidth / 2;
      s.scrollTo({ left, behavior: "smooth" });
    }
  }, [cur]);

  return (
    <div className="shotstrip" ref={strip}>
      {shots.map((s) => {
        const cam = s.camera;
        const fr = s.framing;
        return (
          <button
            key={s.index}
            data-shot={s.index}
            className={`shot ${s.index === cur ? "current" : ""}`}
            onClick={() => onSeek(s.t0 + 0.01)}
            title={`shot ${s.index}: ${fmtTime(s.t0)}–${fmtTime(s.t1)}`}
          >
            {s.thumb ? <img src={api.file(clip, s.thumb)} alt="" loading="lazy" /> : <div className="noimg" />}
            <div className="meta">
              <span className="idx">#{s.index}</span>
              <span className="dim">{fmtTime(s.t0)}</span>
              <span className="dim">{s.duration_s.toFixed(1)} s</span>
            </div>
            <div className="chips">
              {fr?.label && (
                <span className="chip framing" title={`${fr.name} (${Math.round(fr.confidence * 100)}%)`}>
                  {FRAMING_SHORT[fr.label] || fr.label}
                </span>
              )}
              {cam?.label && cam.label !== "unknown" && (
                <span className={`chip cam ${cam.label}`} title={`speed ${cam.speed?.toFixed?.(1)}`}>
                  {cam.label}
                  {cam.direction ? ` ${cam.direction}` : ""}
                </span>
              )}
            </div>
            {s.palette && (
              <div className="pal">
                {s.palette.rgb.map((c, i) => (
                  <span key={i} style={{ background: `rgb(${c.join(",")})`, flexGrow: s.palette.weights[i] }} />
                ))}
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}
