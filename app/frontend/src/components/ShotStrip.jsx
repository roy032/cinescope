import { useEffect, useMemo, useRef, useState } from "react";
import { api, fmtTime } from "../api.js";
import { shotAt } from "../util.js";

const FRAMING_SHORT = { ECS: "ECU", CS: "CU", MS: "MS", FS: "FS", LS: "LS" };

const facets = (s) =>
  [
    s.framing?.label && `framing:${s.framing.label}`,
    s.camera?.label && s.camera.label !== "unknown" && `camera:${s.camera.label}`,
    s.scenes?.type && `scene:${s.scenes.type}`,
  ].filter(Boolean);

// One card per shot: thumbnail, time, framing and camera-move labels, palette.
// Filters keep only the shots of one framing, camera move or scene type.
export default function ShotStrip({ clip, shots, time, onSeek }) {
  const cur = shotAt(shots, time);
  const strip = useRef(null);
  const [filter, setFilter] = useState(null);

  const options = useMemo(() => {
    const c = new Map();
    for (const s of shots) for (const f of facets(s)) c.set(f, (c.get(f) || 0) + 1);
    return [...c.entries()].filter(([, n]) => n < shots.length).sort((a, b) => a[0].localeCompare(b[0]));
  }, [shots]);
  const shown = filter ? shots.filter((s) => facets(s).includes(filter)) : shots;

  useEffect(() => setFilter(null), [clip]);
  useEffect(() => {
    const s = strip.current;
    const el = s?.querySelector(`[data-shot="${cur}"]`);
    if (el) s.scrollTo({ left: el.offsetLeft - s.clientWidth / 2 + el.clientWidth / 2, behavior: "smooth" });
  }, [cur, filter]);

  return (
    <div>
      {options.length > 1 && (
        <div className="filters" role="group" aria-label="filter shots">
          <button className={`chip-btn ${!filter ? "on" : ""}`} onClick={() => setFilter(null)}>
            all {shots.length}
          </button>
          {options.map(([f, n]) => (
            <button key={f} className={`chip-btn ${filter === f ? "on" : ""}`} onClick={() => setFilter(filter === f ? null : f)}>
              {f.split(":")[1]} {n}
            </button>
          ))}
        </div>
      )}
      <div className="shotstrip" ref={strip}>
        {shown.map((s) => {
          const cam = s.camera;
          const fr = s.framing;
          // no thumbnail when the "frames" stage did not run: ask the back end for the middle frame
          const thumb = s.thumb ? api.file(clip, s.thumb) : api.frame(clip, (s.t0 + s.t1) / 2, 320);
          return (
            <button
              key={s.index}
              data-shot={s.index}
              className={`shot ${s.index === cur ? "current" : ""}`}
              onClick={() => onSeek(s.t0 + 0.01, true)}
              title={`Shot ${s.index}: ${fmtTime(s.t0)}–${fmtTime(s.t1)}. Click to play it.`}
              aria-current={s.index === cur ? "true" : undefined}
            >
              <img src={thumb} alt="" loading="lazy" />
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
                  <span className={`chip cam ${cam.label}`} title={`camera: ${cam.label}${cam.direction ? ` ${cam.direction}` : ""}`}>
                    {cam.label}
                    {cam.direction ? ` ${cam.direction}` : ""}
                  </span>
                )}
                {s.scenes?.type && (
                  <span className="chip scene" title={`scene type (${Math.round(100 * s.scenes.probs[s.scenes.type])}%)`}>
                    {s.scenes.type}
                  </span>
                )}
                {s.pose?.max_people > 0 && (
                  <span className="chip people" title={`up to ${s.pose.max_people} people; ${s.pose.events.length} action events`}>
                    {s.pose.max_people} ppl{s.pose.events.length ? ` ${s.pose.events.map((e) => e.type).join(" ")}` : ""}
                  </span>
                )}
              </div>
              {s.cutouts?.file && <img className="cut" src={api.file(clip, s.cutouts.file)} alt="cut-out" loading="lazy" />}
              {s.palette && (
                <div className="pal" title="the shot's palette">
                  {s.palette.rgb.map((c, i) => (
                    <span key={i} style={{ background: `rgb(${c.join(",")})`, flexGrow: s.palette.weights[i] }} />
                  ))}
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
