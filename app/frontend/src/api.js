// Thin wrapper over the FastAPI back end (same origin; the Vite dev server proxies /api).

async function get(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.json();
}

async function post(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.json();
}

const enc = encodeURIComponent;

export const api = {
  clips: () => get("/api/clips"),
  stages: () => get("/api/stages"),
  report: (clip) => get(`/api/clips/${enc(clip)}/report`),
  startJob: (clip, stages, backend) => post("/api/jobs", { clip, stages, backend }),
  job: (id) => get(`/api/jobs/${id}`),
  cancel: (id) => post(`/api/jobs/${id}/cancel`),
  search: (clip, q, k = 8) => get(`/api/clips/${enc(clip)}/search?q=${enc(q)}&k=${k}`),
  ask: (clip, q) => get(`/api/clips/${enc(clip)}/ask?q=${enc(q)}`),
  campath: (clip) => get(`/api/clips/${enc(clip)}/campath`),
  upload: async (file) => {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/api/clips/upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
    return r.json();
  },
  file: (clip, rel) => `/api/clips/${enc(clip)}/files/${rel}`,
  frame: (clip, t, width = 640) => `/api/clips/${enc(clip)}/frame?t=${t.toFixed(2)}&width=${width}`,
  video: (clip) => `/api/clips/${enc(clip)}/video`,
};

export function fmtTime(t) {
  const m = Math.floor(t / 60);
  const s = t - 60 * m;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}
