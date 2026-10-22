// Thin wrapper over the FastAPI back end (same origin; the Vite dev server proxies /api).
export { fmtTime } from "./util.js";

// An HTTP error that keeps its status, so callers can tell "not processed yet" (404)
// from a real failure. The message is FastAPI's `detail` when there is one.
export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `HTTP ${status}`);
    this.status = status;
  }
}

async function check(r) {
  if (r.ok) return r.json();
  const text = await r.text();
  let detail = text.slice(0, 300);
  try {
    const d = JSON.parse(text).detail;
    if (d) detail = typeof d === "string" ? d : JSON.stringify(d);
  } catch {
    /* not JSON */
  }
  throw new ApiError(r.status, detail);
}

async function get(path, opts) {
  return check(await fetch(path, opts));
}

async function post(path, body) {
  return check(
    await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    }),
  );
}

const enc = encodeURIComponent;

export const api = {
  clips: () => get("/api/clips"),
  stages: () => get("/api/stages"),
  report: (clip) => get(`/api/clips/${enc(clip)}/report`),
  startJob: (clip, stages, backend) => post("/api/jobs", { clip, stages, backend }),
  job: (id) => get(`/api/jobs/${id}`),
  jobs: (clip) => get(clip ? `/api/jobs?clip=${enc(clip)}` : "/api/jobs"),
  cancel: (id) => post(`/api/jobs/${id}/cancel`),
  search: (clip, q, k = 8, signal) => get(`/api/clips/${enc(clip)}/search?q=${enc(q)}&k=${k}`, { signal }),
  ask: (clip, q, signal) => get(`/api/clips/${enc(clip)}/ask?q=${enc(q)}`, { signal }),
  campath: (clip) => get(`/api/clips/${enc(clip)}/campath`),
  cutout: (clip, t, method = "grabcut") => `/api/clips/${enc(clip)}/cutout?t=${t.toFixed(2)}&method=${method}`,
  // XHR rather than fetch: fetch cannot report upload progress, and films are gigabytes.
  upload: (file, onProgress) =>
    new Promise((resolve, reject) => {
      const fd = new FormData();
      fd.append("file", file);
      const x = new XMLHttpRequest();
      x.open("POST", "/api/clips/upload");
      x.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
      x.onload = () => {
        let body = null;
        try {
          body = JSON.parse(x.responseText);
        } catch {
          /* not JSON */
        }
        if (x.status >= 200 && x.status < 300) resolve(body);
        else reject(new ApiError(x.status, body?.detail || x.responseText.slice(0, 300)));
      };
      x.onerror = () => reject(new ApiError(0, "upload failed: the server could not be reached"));
      x.send(fd);
    }),
  file: (clip, rel) => `/api/clips/${enc(clip)}/files/${rel}`,
  frame: (clip, t, width = 640) => `/api/clips/${enc(clip)}/frame?t=${t.toFixed(2)}&width=${width}`,
  video: (clip) => `/api/clips/${enc(clip)}/video`,
  // editing-software hand-off: edl | fcpxml | otio | csv | vtt | pdf (a download unless inline)
  exportUrl: (clip, fmt, inline = false) => `/api/clips/${enc(clip)}/export/${fmt}${inline ? "?download=false" : ""}`,
  libraryUrl: () => "/api/library",
  compareUrl: (a, b) => `/api/compare?a=${enc(a)}&b=${enc(b)}&format=html`,
};
