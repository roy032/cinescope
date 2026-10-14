// Per-stage wall-clock time of the run that produced this report, and its
// extrapolation to a 2-hour film (time scales with video length).
export default function Timing({ report }) {
  const t = report.timing;
  if (!t) return null;
  const rows = Object.entries(t.stages);
  const max = Math.max(...rows.map(([, s]) => s.total_s || 0), 1e-6);
  const h2 = t.extrapolated_2h_film_s;
  return (
    <div className="timing">
      <p className="dim small">
        backend <b>{t.backend}</b> · {t.workers} worker{t.workers > 1 ? "s" : ""} · {t.threads} threads ·{" "}
        {t.cpu_count} CPUs · wall {t.wall_s.toFixed(1)} s ({t.video_s_per_wall_s}× real time)
        {h2 ? ` · a 2-hour film at this rate: ${(h2 / 3600).toFixed(1)} h` : ""}
      </p>
      <table>
        <tbody>
          {rows.map(([name, s]) => (
            <tr key={name}>
              <td className="stage">{name}</td>
              <td className="barcell">
                <div className="tbar" style={{ width: `${(100 * (s.total_s || 0)) / max}%` }} />
              </td>
              <td className="num">{s.cached ? "cached" : `${(s.total_s || 0).toFixed(1)} s`}</td>
              <td className="num dim">{!s.cached && s.video_s_per_wall_s ? `${s.video_s_per_wall_s}×` : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
