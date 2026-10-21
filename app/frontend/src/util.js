// Pure helpers (no DOM, no fetch), so `npm test` can run them under plain Node.

// 75.3 -> "1:15.3"; a film's 7384.2 -> "2:03:04.2"
export function fmtTime(t) {
  const v = Math.max(0, Number(t) || 0);
  const h = Math.floor(v / 3600);
  const m = Math.floor((v - 3600 * h) / 60);
  const s = v - 3600 * h - 60 * m;
  const ss = s.toFixed(1).padStart(4, "0");
  // 59.96 s rounds to "60.0": carry it rather than print 0:60.0
  if (ss === "60.0") return fmtTime(Math.ceil(v));
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

// Seconds of work as people say them: 0.4 s, 42 s, 3 min 5 s, 1 h 20 min.
export function fmtDuration(s) {
  const v = Math.max(0, Number(s) || 0);
  if (v < 10) return `${v.toFixed(1)} s`;
  if (v < 60) return `${Math.round(v)} s`;
  if (v < 3600) {
    const m = Math.floor(v / 60);
    const r = Math.round(v - 60 * m);
    return r ? `${m} min ${r} s` : `${m} min`;
  }
  const h = Math.floor(v / 3600);
  const m = Math.round((v - 3600 * h) / 60);
  return m ? `${h} h ${m} min` : `${h} h`;
}

// 1.9338 -> "1.9×", 71.926 -> "72×", 0.042 -> "0.04×"
export function fmtSpeed(x) {
  const v = Number(x) || 0;
  if (v >= 10) return `${Math.round(v)}×`;
  if (v >= 1) return `${v.toFixed(1)}×`;
  return `${v.toFixed(2)}×`;
}

export function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

// The shot playing at time t (shots are [t0, t1) in order), or -1.
export function shotAt(shots, t) {
  let lo = 0;
  let hi = shots.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (t < shots[mid].t0) hi = mid - 1;
    else if (t >= shots[mid].t1) lo = mid + 1;
    else return mid;
  }
  return -1;
}

// Shot statistics a film editor would quote: count, average shot length (ASL),
// median, longest, cuts per minute, and how often each framing and camera move occurs.
export function shotStats(shots, duration) {
  const d = shots.map((s) => s.duration_s).filter((x) => x > 0);
  if (!d.length) return null;
  const sorted = [...d].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  const longest = shots.reduce((a, s) => (s.duration_s > a.duration_s ? s : a), shots[0]);
  const count = (key) => {
    const c = new Map();
    for (const s of shots) {
      const v = key(s);
      if (v) c.set(v, (c.get(v) || 0) + (s.duration_s || 0));
    }
    const total = [...c.values()].reduce((a, b) => a + b, 0) || 1;
    return [...c.entries()].map(([label, secs]) => ({ label, secs, frac: secs / total })).sort((a, b) => b.secs - a.secs);
  };
  const len = duration || d.reduce((a, b) => a + b, 0);
  return {
    n: shots.length,
    asl: d.reduce((a, b) => a + b, 0) / d.length,
    median,
    longest,
    cutsPerMin: len > 0 ? (60 * Math.max(shots.length - 1, 0)) / len : 0,
    framing: count((s) => s.framing?.name || s.framing?.label),
    camera: count((s) => (s.camera?.label && s.camera.label !== "unknown" ? s.camera.label : null)),
    scenes: count((s) => s.scenes?.type),
  };
}

// The steps of a job as a checklist: done / running / cached / waiting.
// `plan` is the back end's run order ("decode" first); a step that never reported
// before a later one did was loaded from the cache (or had nothing to decode).
export function jobSteps(job) {
  const plan = job?.plan || [];
  const seen = job?.stage_progress || {};
  const finished = job && !["queued", "running"].includes(job.status);
  const last = Math.max(-1, ...plan.map((s, i) => (s in seen ? i : -1)));
  return plan.map((name, i) => {
    const p = seen[name];
    let state;
    if (p >= 1) state = "done";
    else if (p !== undefined) state = finished ? (job.status === "done" ? "done" : "stopped") : "running";
    else if (i < last || (finished && job.status === "done")) state = "cached";
    else state = finished ? "skipped" : "waiting";
    return { name, state, progress: p ?? 0 };
  });
}

// Estimated seconds left from elapsed time and progress, once there is enough to go on.
export function eta(elapsed, progress) {
  if (!(progress > 0.03) || !(elapsed > 3) || progress >= 1) return null;
  return (elapsed * (1 - progress)) / progress;
}

// "?clip=a&t=12.3" <-> {clip, t}
export function parseLocation(search) {
  const p = new URLSearchParams(search);
  const t = Number(p.get("t"));
  return { clip: p.get("clip") || "", t: Number.isFinite(t) && t > 0 ? t : 0 };
}

// A copy of `obj` without `keys` (timelines that have a panel of their own).
export function omitKeys(obj, keys) {
  if (!obj) return obj;
  const out = Object.fromEntries(Object.entries(obj).filter(([k]) => !keys.includes(k)));
  return Object.keys(out).length ? out : null;
}

// Colour temperature (Lab warmth, about -30..30) as a diverging colour: blue for
// cool, neutral grey at 0, amber for warm.
export function warmthColor(w) {
  const x = clamp((Number(w) || 0) / 20, -1, 1);
  const mix = (a, b) => a.map((v, i) => Math.round(v + (b[i] - v) * Math.abs(x)));
  const grey = [128, 128, 128];
  const [r, g, b] = mix(grey, x < 0 ? [64, 156, 214] : [235, 150, 50]);
  return `rgb(${r}, ${g}, ${b})`;
}

// Luma 0..1 as a grey.
export function lumaColor(y) {
  const v = Math.round(255 * clamp(Number(y) || 0, 0, 1));
  return `rgb(${v}, ${v}, ${v})`;
}
