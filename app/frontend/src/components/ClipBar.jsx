import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

// Pick or upload a clip, choose stages and start a job; shows the job's progress.
export default function ClipBar({ clips, clip, onClip, onDone, onUploaded }) {
  const [stages, setStages] = useState([]);
  const [chosen, setChosen] = useState(null); // null = all stages
  const [open, setOpen] = useState(false);
  const [backend, setBackend] = useState("onnx");
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const fileRef = useRef(null);

  useEffect(() => {
    api.stages().then(setStages).catch(() => setStages([]));
  }, []);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return undefined;
    const id = setInterval(async () => {
      try {
        const j = await api.job(job.id);
        setJob(j);
        if (j.status === "done") onDone();
      } catch {
        /* the server restarted: stop polling */
        setJob(null);
      }
    }, 1000);
    return () => clearInterval(id);
  }, [job, onDone]);

  const start = async () => {
    setBusy(true);
    try {
      setJob(await api.startJob(clip, chosen, backend));
    } catch (e) {
      setJob({ status: "error", error: String(e) });
    } finally {
      setBusy(false);
    }
  };

  const upload = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setBusy(true);
    try {
      const r = await api.upload(f);
      onUploaded(r.clip);
    } catch (err) {
      setJob({ status: "error", error: String(err) });
    } finally {
      setBusy(false);
      e.target.value = "";
    }
  };

  const running = job && ["queued", "running"].includes(job.status);
  const toggle = (n) => {
    const all = stages.map((s) => s.name);
    const cur = new Set(chosen ?? all);
    if (cur.has(n)) cur.delete(n);
    else cur.add(n);
    setChosen(cur.size === all.length ? null : all.filter((x) => cur.has(x)));
  };

  return (
    <div className="clipbar">
      <select value={clip} onChange={(e) => onClip(e.target.value)} aria-label="clip">
        {clips.map((c) => (
          <option key={c.clip} value={c.clip}>
            {c.processed ? "● " : "○ "}
            {c.clip} ({c.size_mb} MB)
          </option>
        ))}
      </select>
      <button className="ghost" onClick={() => fileRef.current?.click()} disabled={busy}>
        Upload…
      </button>
      <input ref={fileRef} type="file" accept="video/*,.mkv,.avi" hidden onChange={upload} />
      <button className="ghost" onClick={() => setOpen(!open)} aria-expanded={open}>
        Stages {chosen ? `(${chosen.length})` : "(all)"}
      </button>
      <button className="primary" onClick={start} disabled={!clip || busy || running}>
        {running ? "Analysing…" : "Analyse"}
      </button>
      {running && (
        <button className="ghost" onClick={() => api.cancel(job.id).then(setJob)}>
          Stop
        </button>
      )}
      {job && (
        <div className={`job ${job.status}`}>
          <div className="bar">
            <div style={{ width: `${Math.round((job.progress || 0) * 100)}%` }} />
          </div>
          <span>
            {job.status === "running" ? `${job.stage} ${job.message || ""}` : job.status}
            {job.error ? `: ${job.error}` : ""}
          </span>
        </div>
      )}
      {open && (
        <div className="stage-menu">
          <label className="backend">
            backend
            <select value={backend} onChange={(e) => setBackend(e.target.value)}>
              <option value="onnx">onnx</option>
              <option value="torch">torch</option>
              <option value="tensorrt">tensorrt (GPU, untested)</option>
            </select>
          </label>
          {stages.map((s) => (
            <label key={s.name} title={`requires: ${s.requires.join(", ") || "-"}`}>
              <input type="checkbox" checked={!chosen || chosen.includes(s.name)} onChange={() => toggle(s.name)} />
              {s.name} <small>{s.kind}</small>
            </label>
          ))}
          <p className="hint">Required stages are added automatically; cached ones are not recomputed.</p>
        </div>
      )}
    </div>
  );
}
