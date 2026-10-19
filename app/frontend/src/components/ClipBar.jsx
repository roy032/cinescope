import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useDismiss } from "../hooks.js";
import { fmtTime } from "../util.js";

const BACKEND_KEY = "cinescope.backend";

function savedBackend() {
  try {
    return localStorage.getItem(BACKEND_KEY) || "onnx";
  } catch {
    return "onnx";
  }
}

// Pick or upload a clip, choose stages and the model backend, and start the analysis.
export default function ClipBar({ clips, clip, onClip, job, onStart, onUpload, uploading }) {
  const [stages, setStages] = useState([]);
  const [chosen, setChosen] = useState(null); // null = the default stages (all but the slow optional ones)
  const [open, setOpen] = useState(false);
  const [backend, setBackend] = useState(savedBackend);
  const fileRef = useRef(null);
  const menuRef = useRef(null);
  useDismiss(open, setOpen, menuRef);

  useEffect(() => {
    api.stages().then(setStages).catch(() => setStages([]));
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(BACKEND_KEY, backend);
    } catch {
      /* private window */
    }
  }, [backend]);

  const row = clips.find((c) => c.clip === clip);
  const done = new Set(row?.stages || []);
  const running = job.active;
  const all = stages.map((s) => s.name);
  const defaults = stages.filter((s) => s.default !== false).map((s) => s.name);
  const picked = chosen ?? defaults;
  const toggle = (n) => {
    const cur = new Set(picked);
    if (cur.has(n)) cur.delete(n);
    else cur.add(n);
    const next = all.filter((x) => cur.has(x));
    setChosen(next.length === defaults.length && next.every((x) => defaults.includes(x)) ? null : next);
  };
  const start = () => {
    setOpen(false);
    onStart(chosen, backend);
  };

  const analysed = clips.filter((c) => c.processed);
  const fresh = clips.filter((c) => !c.processed);
  const label = (c) =>
    `${c.clip}${c.duration_s ? ` · ${fmtTime(c.duration_s)}` : ""} · ${c.size_mb >= 1024 ? `${(c.size_mb / 1024).toFixed(1)} GB` : `${c.size_mb} MB`}`;

  return (
    <div className="clipbar">
      {clips.length > 0 && (
        <select value={clip} onChange={(e) => onClip(e.target.value)} aria-label="clip" className="clip-select">
          {analysed.length > 0 && (
            <optgroup label="Analysed">
              {analysed.map((c) => (
                <option key={c.clip} value={c.clip}>
                  {label(c)}
                </option>
              ))}
            </optgroup>
          )}
          {fresh.length > 0 && (
            <optgroup label="Not analysed yet">
              {fresh.map((c) => (
                <option key={c.clip} value={c.clip}>
                  {label(c)}
                </option>
              ))}
            </optgroup>
          )}
        </select>
      )}
      <button className="ghost" onClick={() => fileRef.current?.click()} disabled={uploading != null} title="Or drop a video anywhere on the page">
        {uploading != null ? `Uploading ${Math.round(uploading * 100)}%` : "Upload…"}
      </button>
      <input
        ref={fileRef}
        type="file"
        accept="video/*,.mkv,.avi"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (f) onUpload(f);
        }}
      />
      {clip && (
        <div className="menu-wrap" ref={menuRef}>
          <button className="ghost" onClick={() => setOpen(!open)} aria-expanded={open} aria-haspopup="dialog">
            Stages {chosen ? `(${chosen.length} of ${all.length})` : "(default)"} ▾
          </button>
          {open && (
            <div className="stage-menu" role="dialog" aria-label="stages to run">
              <label className="backend">
                Model backend
                <select value={backend} onChange={(e) => setBackend(e.target.value)}>
                  <option value="onnx">ONNX Runtime (fastest on a CPU)</option>
                  <option value="torch">PyTorch</option>
                  <option value="tensorrt">TensorRT (NVIDIA GPU)</option>
                </select>
              </label>
              <div className="quick">
                <button className="link" onClick={() => setChosen(null)}>
                  default
                </button>
                <button className="link" onClick={() => setChosen(all)}>
                  everything
                </button>
                <button className="link" onClick={() => setChosen(all.filter((n) => ["shots", "palette", "camera"].includes(n)))}>
                  quick (shots, palette, camera)
                </button>
                <button className="link" onClick={() => setChosen(all.filter((n) => !done.has(n)))} disabled={!done.size}>
                  not run yet
                </button>
              </div>
              {stages.map((s) => (
                <label key={s.name} title={`needs: ${s.requires.join(", ") || "nothing"}`}>
                  <input type="checkbox" checked={picked.includes(s.name)} onChange={() => toggle(s.name)} />
                  {s.name}
                  {done.has(s.name) ? <small className="ok"> ✓ done</small> : <small>{s.default === false ? "optional, slow" : ""}</small>}
                </label>
              ))}
              <p className="hint">Stages a choice needs are added automatically; finished ones load from the cache.</p>
            </div>
          )}
        </div>
      )}
      {clip && (
        <button className="primary" onClick={start} disabled={running}title={row?.processed ? "Run again; finished stages are reused" : ""}>
          {running ? "Analysing…" : row?.processed ? "Re-analyse" : "Analyse"}
        </button>
      )}
      {running && job.job?.status === "running" && (
        <span className="mini-progress" title={`${job.job.stage} ${job.job.message || ""}`}>
          <span style={{ width: `${Math.round((job.job.progress || 0) * 100)}%` }} />
        </span>
      )}
    </div>
  );
}
