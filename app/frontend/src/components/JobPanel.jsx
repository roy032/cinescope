import { eta, fmtDuration, jobSteps } from "../util.js";

const MARK = { done: "✓", cached: "✓", running: "", waiting: "", skipped: "–", stopped: "■", failed: "✕" };
const TITLE = {
  queued: "Waiting for the previous analysis to finish…",
  running: "Analysing",
  done: "Analysis finished",
  cancelled: "Analysis stopped",
  error: "Analysis failed",
};

// The progress of an analysis job: an overall bar, time left, and one line per step.
export default function JobPanel({ job, onCancel, onDismiss }) {
  if (!job) return null;
  const steps = jobSteps(job);
  const active = ["queued", "running"].includes(job.status);
  const left = active ? eta(job.elapsed_s, job.progress) : null;
  const pct = Math.round((job.status === "done" ? 1 : job.progress || 0) * 100);
  const nFailed = Object.keys(job.errors || {}).length;
  const title = job.status === "done" && nFailed ? `Analysis finished, ${nFailed} stage${nFailed > 1 ? "s" : ""} failed` : TITLE[job.status] || job.status;
  return (
    <div className={`panel jobpanel ${job.status}${nFailed ? " partial" : ""}`} role="status" aria-live="polite">
      <div className="job-head">
        <div>
          <h2>{title}</h2>
          <div className="dim small">
            {job.status === "running" && job.stage && (
              <>
                <b className="fg">{job.stage}</b> {job.message}
                {" · "}
              </>
            )}
            {job.elapsed_s > 0 && `${fmtDuration(job.elapsed_s)} elapsed`}
            {left != null && ` · about ${fmtDuration(left)} left`}
            {job.backend && ` · ${job.backend} backend`}
          </div>
        </div>
        <div className="job-actions">
          {active && onCancel && (
            <button className="ghost" onClick={onCancel} title="Stops at the next frame; a new run resumes from the last checkpoint">
              Stop
            </button>
          )}
          {!active && onDismiss && (
            <button className="ghost" onClick={onDismiss} aria-label="dismiss">
              Close
            </button>
          )}
        </div>
      </div>
      <div className="bigbar" aria-label={`${pct}% done`}>
        <div style={{ width: `${pct}%` }} />
      </div>
      {job.error && <div className="error small">{job.error}</div>}
      {steps.length > 0 && (
        <ol className="steps">
          {steps.map((s) => (
            <li key={s.name} className={`step ${s.state}`} title={s.state}>
              <span className="mark" aria-hidden="true">
                {MARK[s.state]}
              </span>
              <span className="sname">{s.name === "decode" ? "decode frames" : s.name}</span>
              {s.state === "running" && (
                <span className="sbar">
                  <span style={{ width: `${Math.round(s.progress * 100)}%` }} />
                </span>
              )}
              {s.state === "cached" && <span className="dim small">cached</span>}
              {s.note && (
                <span className={`small snote${s.state === "failed" ? " bad" : " dim"}`} title={s.note}>
                  {s.note}
                </span>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
