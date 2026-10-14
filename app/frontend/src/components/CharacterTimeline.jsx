import { api, fmtTime } from "../api.js";

// One row per character: face, on-screen spans (bars) and talking spans (marks on top).
export default function CharacterTimeline({ clip, chars, duration, time, onSeek, maxRows = 12 }) {
  if (!chars.length) return <p className="dim">No faces were tracked long enough to count as characters.</p>;
  const rows = chars.slice(0, maxRows);
  const W = 1000;
  const x = (t) => (W * t) / Math.max(duration, 1e-3);
  const click = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    onSeek(((e.clientX - r.left) / r.width) * duration);
  };
  return (
    <div className="chars">
      {rows.map((c) => (
        <div className="char-row" key={c.id}>
          <div className="who">
            <img src={api.file(clip, c.thumb)} alt={c.id} />
            <div>
              <b>{c.id}</b>
              <div className="dim small">
                {c.screen_time_s.toFixed(1)} s on screen
                <br />
                {c.speaking_time_s.toFixed(1)} s talking
              </div>
            </div>
          </div>
          <svg viewBox={`0 0 ${W} 28`} preserveAspectRatio="none" className="lane" onClick={click}>
            <rect x="0" y="0" width={W} height="28" className="lane-bg" />
            {c.on_screen.map(([a, b], i) => (
              <rect key={`o${i}`} x={x(a)} y="6" width={Math.max(x(b) - x(a), 1.5)} height="16" className="on">
                <title>{`on screen ${fmtTime(a)}–${fmtTime(b)}`}</title>
              </rect>
            ))}
            {c.speaking.map(([a, b], i) => (
              <rect key={`s${i}`} x={x(a)} y="10" width={Math.max(x(b) - x(a), 1.5)} height="8" className="talk">
                <title>{`talking ${fmtTime(a)}–${fmtTime(b)}`}</title>
              </rect>
            ))}
            <line x1={x(time)} x2={x(time)} y1="0" y2="28" className="ph" />
          </svg>
        </div>
      ))}
      <div className="legend">
        <span className="sw on" /> on screen <span className="sw talk" /> talking
        {chars.length > maxRows && <span className="dim"> · {chars.length - maxRows} more characters not shown</span>}
      </div>
    </div>
  );
}
