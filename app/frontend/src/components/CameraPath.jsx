import { useEffect, useMemo, useRef, useState } from "react";
import { api, fmtTime } from "../api.js";

// Phase 5's camera path for one shot: camera centres (a line), the sparse 3D
// points when SfM succeeded, and a small frustum at the current time. A plain
// orbit view on a canvas (drag to turn); no 3D library needed for a polyline.
export default function CameraPath({ clip, info, time, onSeek }) {
  const [doc, setDoc] = useState(null);
  const [err, setErr] = useState("");
  const [view, setView] = useState({ yaw: -0.6, pitch: 0.35 });
  const canvas = useRef(null);
  const drag = useRef(null);

  useEffect(() => {
    setDoc(null);
    api.campath(clip).then(setDoc).catch((e) => setErr(String(e)));
  }, [clip, info.file]);

  // centre and scale the scene once
  const scene = useMemo(() => {
    if (!doc?.frames?.length) return null;
    const cs = doc.frames.map((f) => f.centre);
    const pts = doc.points || [];
    const all = cs.concat(pts);
    const mean = [0, 1, 2].map((k) => all.reduce((a, p) => a + p[k], 0) / all.length);
    const rad = Math.max(...all.map((p) => Math.hypot(p[0] - mean[0], p[1] - mean[1], p[2] - mean[2])), 1e-6);
    return { mean, rad };
  }, [doc]);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !doc || !scene) return;
    const W = (c.width = c.clientWidth * (window.devicePixelRatio || 1));
    const H = (c.height = c.clientHeight * (window.devicePixelRatio || 1));
    const g = c.getContext("2d");
    const css = getComputedStyle(c);
    g.clearRect(0, 0, W, H);
    const { yaw, pitch } = view;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const s = (0.42 * Math.min(W, H)) / scene.rad;
    // OpenCV camera axes: x right, y down, z forward -> draw with y up
    const proj = (p) => {
      const x = p[0] - scene.mean[0], y = -(p[1] - scene.mean[1]), z = p[2] - scene.mean[2];
      const x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
      const y2 = cp * y - sp * z1;
      return [W / 2 + s * x1, H / 2 - s * y2];
    };
    g.fillStyle = css.getPropertyValue("--dim") || "#888";
    for (const p of doc.points || []) {
      const [u, v] = proj(p);
      g.fillRect(u, v, 1.5, 1.5);
    }
    g.strokeStyle = css.getPropertyValue("--accent") || "#e8a33d";
    g.lineWidth = 2 * (window.devicePixelRatio || 1);
    g.beginPath();
    doc.frames.forEach((f, i) => {
      const [u, v] = proj(f.centre);
      if (i === 0) g.moveTo(u, v);
      else g.lineTo(u, v);
    });
    g.stroke();
    // current camera: nearest path frame to the playhead, drawn as a frustum
    let k = 0;
    doc.frames.forEach((f, i) => {
      if (Math.abs(f.t - time) < Math.abs(doc.frames[k].t - time)) k = i;
    });
    const f = doc.frames[k];
    const R = f.R; // world -> camera; camera axes in world are the rows
    const d = scene.rad * 0.18;
    const corner = (a, b) => [0, 1, 2].map((j) => f.centre[j] + d * (R[2][j] + 0.6 * a * R[0][j] + 0.4 * b * R[1][j]));
    const o = proj(f.centre);
    const cs = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)].map(proj);
    const inShot = time >= doc.t0 && time <= doc.t1;
    g.strokeStyle = inShot ? css.getPropertyValue("--talk") || "#ff5a5a" : css.getPropertyValue("--fg") || "#ddd";
    g.lineWidth = 1.5 * (window.devicePixelRatio || 1);
    g.beginPath();
    for (const q of cs) {
      g.moveTo(o[0], o[1]);
      g.lineTo(q[0], q[1]);
    }
    g.moveTo(cs[3][0], cs[3][1]);
    for (const q of cs) g.lineTo(q[0], q[1]);
    g.stroke();
  }, [doc, scene, view, time]);

  if (err) return <p className="dim">{err}</p>;
  if (!doc) return <p className="dim">loading…</p>;

  const down = (e) => {
    drag.current = { x: e.clientX, y: e.clientY, ...view };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const move = (e) => {
    if (!drag.current) return;
    const d = drag.current;
    setView({
      yaw: d.yaw + (e.clientX - d.x) * 0.01,
      pitch: Math.max(-1.5, Math.min(1.5, d.pitch + (e.clientY - d.y) * 0.01)),
    });
  };
  const last = doc.frames[doc.frames.length - 1];
  const pan = last ? last.pan_deg : 0;
  const tilt = last ? last.tilt_deg : 0;
  return (
    <div className="campath">
      <canvas
        ref={canvas}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={() => (drag.current = null)}
        onDoubleClick={() => onSeek(doc.t0 + 0.01)}
        aria-label="camera path, drag to turn, double-click to play the shot"
      />
      <div className="dim small">
        {doc.frames.length} poses{doc.points?.length ? `, ${doc.points.length} 3D points` : ""} · {fmtTime(doc.t0)}–
        {fmtTime(doc.t1)} · total pan {pan.toFixed(1)}°, tilt {tilt.toFixed(1)}° · double-click to play the shot
      </div>
    </div>
  );
}
