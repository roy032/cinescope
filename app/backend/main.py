"""CineScope web app, back end: FastAPI over cinescope.pipeline.

    uvicorn app.backend.main:app --port 8000          # from the repo root
    # then open http://localhost:8000 (the built front end, app/frontend/dist, if present)

Endpoints (all JSON unless noted):

    GET  /api/health
    GET  /api/stages                        the registered pipeline stages
    GET  /api/clips                         clips found in the clip folders, and whether processed
    POST /api/clips/upload                  multipart file -> saved to the uploads folder
    POST /api/jobs                          {"clip", "stages"?, "backend"?, "params"?} -> job
    GET  /api/jobs, /api/jobs/{id}          status, current stage, progress 0..1
    POST /api/jobs/{id}/cancel              stops at the next frame (the decode checkpoint stays:
                                            a new job on the clip resumes there)
    GET  /api/clips/{clip}/report           outputs/<clip>/report.json
    GET  /api/clips/{clip}/files/{path}     any asset under outputs/<clip>/ (thumbnails, barcode ...)
    GET  /api/clips/{clip}/frame?t=&width=  one frame as JPEG, decoded on demand
    GET  /api/clips/{clip}/video            the video file (HTTP range requests, for <video>)
    GET  /api/clips/{clip}/search?q=&k=     Phase 8 scene search over the clip's index
    GET  /api/clips/{clip}/ask?q=           Phase 8 "chat with the movie" (template answer, LLM optional)
    GET  /api/clips/{clip}/campath          the camera path of one shot (Phase 5)

Configuration (environment):
    CINESCOPE_CLIPS    folders to list clips from, separated by the OS path separator
                       (default: trailers/, data/clips/ and the uploads folder)
    CINESCOPE_OUT      where results go (default outputs/app)
    CINESCOPE_BACKEND  default model backend for jobs: torch | onnx | tensorrt (default onnx)
    CINESCOPE_WEIGHTS  model weights (default data/weights)

Jobs run one at a time in a background thread: the stages already use the
CPU's cores, and two pipelines at once would only fight over them.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover - the CLI pipeline does not need the web app
    raise ImportError("the web app needs: pip install fastapi uvicorn python-multipart") from e


class JobRequest(BaseModel):
    clip: str
    stages: list[str] | None = None
    backend: str | None = None
    params: dict[str, dict] | None = None
    workers: int | None = None


@dataclass
class Job:
    id: str
    clip: str
    stages: list[str] | None
    backend: str
    status: str = "queued"            # queued | running | done | error | cancelled
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    stage_progress: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["elapsed_s"] = round((self.finished or time.time()) - self.started, 1) if self.started else 0.0
        return d


def _safe_join(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if root.resolve() not in p.parents and p != root.resolve():
        raise HTTPException(404, "not found")
    return p


def create_app(clip_dirs: list[str | Path] | None = None, out_dir: str | Path | None = None,
               default_backend: str | None = None, encoder_factory=None, runner=None,
               frontend_dist: str | Path | None = None) -> FastAPI:
    """encoder_factory(spec) -> encoder and runner(**kw) -> report are injectable for tests."""
    out = Path(out_dir or os.environ.get("CINESCOPE_OUT", ROOT / "outputs" / "app"))
    out.mkdir(parents=True, exist_ok=True)
    uploads = out / "_uploads"
    uploads.mkdir(exist_ok=True)
    if clip_dirs is None:
        env = os.environ.get("CINESCOPE_CLIPS")
        clip_dirs = env.split(os.pathsep) if env else [ROOT / "trailers", ROOT / "data" / "clips"]
    dirs = [Path(d) for d in clip_dirs] + [uploads]
    backend = default_backend or os.environ.get("CINESCOPE_BACKEND", "onnx")
    dist = Path(frontend_dist) if frontend_dist else ROOT / "app" / "frontend" / "dist"

    jobs: dict[str, Job] = {}
    cancel: set[str] = set()
    queue_lock = threading.Lock()
    run_lock = threading.Lock()

    app = FastAPI(title="CineScope", version="0.9")

    # ------------------------------------------------------------------ clips
    def clips() -> dict[str, Path]:
        found: dict[str, Path] = {}
        for d in dirs:
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in VIDEO_EXT and p.stem not in found:
                    found[p.stem] = p
        return found

    def clip_path(clip: str) -> Path:
        p = clips().get(clip)
        if p is None:
            raise HTTPException(404, f"no clip {clip!r}")
        return p

    def report_path(clip: str) -> Path:
        return out / clip / "report.json"

    def load_report(clip: str) -> dict:
        p = report_path(clip)
        if not p.exists():
            raise HTTPException(404, f"{clip!r} has not been processed yet")
        return json.loads(p.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health():
        return {"ok": True, "backend": backend, "clip_dirs": [str(d) for d in dirs]}

    @app.get("/api/stages")
    def stages():
        from cinescope.pipeline import STAGES
        from cinescope.pipeline import stages as _builtin  # noqa: F401

        return [{"name": n, "version": c.version, "requires": list(c.requires),
                 "kind": "frame" if c.streams else "post", "params": c.defaults} for n, c in STAGES.items()]

    @app.get("/api/clips")
    def list_clips():
        rows = []
        for name, p in clips().items():
            rp = report_path(name)
            row = {"clip": name, "file": p.name, "size_mb": round(p.stat().st_size / 2**20, 1),
                   "uploaded": p.parent == uploads, "processed": rp.exists()}
            if rp.exists():
                try:
                    r = json.loads(rp.read_text(encoding="utf-8"))
                    row.update(duration_s=r["info"]["duration_s"], stages=r.get("stages", []),
                               n_shots=len(r.get("shots", [])))
                except (OSError, json.JSONDecodeError, KeyError):
                    pass
            rows.append(row)
        return rows

    @app.post("/api/clips/upload")
    async def upload(file: UploadFile = File(...)):  # noqa: B008 - FastAPI's way
        name = Path(file.filename or "upload.mp4").name
        if Path(name).suffix.lower() not in VIDEO_EXT:
            raise HTTPException(400, f"not a video file: {name}")
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
        dest = uploads / safe
        with open(dest, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        return {"clip": dest.stem, "file": dest.name, "size_mb": round(dest.stat().st_size / 2**20, 1)}

    # ------------------------------------------------------------------ jobs
    def work(job: Job, req: JobRequest) -> None:
        from cinescope.pipeline import Interrupted, Options, run

        with run_lock:                                   # one pipeline at a time
            if job.id in cancel:
                job.status = "cancelled"
                return
            job.status, job.started = "running", time.time()

            def progress(stage: str, frac: float, msg: str) -> None:
                job.stage, job.message = stage, msg
                job.stage_progress[stage] = round(frac, 3)
                done = sum(1 for v in job.stage_progress.values() if v >= 1.0)
                n = max(len(job.stage_progress), 1)
                job.progress = round(min(0.99, (done + (0 if frac >= 1 else frac)) / (n + 1)), 3)

            def stop(n: int, t: float) -> bool:
                return job.id in cancel

            try:
                opts = Options(backend=req.backend or backend, workers=req.workers or 1,
                               threads=int(os.environ.get("OMP_NUM_THREADS", 2) or 2))
                fn = runner or run
                fn(video=str(clip_path(req.clip)), stages=req.stages, out=out, params=req.params or {},
                   options=opts, progress=progress, log=None, stop=stop)
                job.status, job.progress, job.stage = "done", 1.0, "done"
            except Interrupted:
                job.status = "cancelled"
            except Exception as e:                       # noqa: BLE001 - reported to the client
                job.status, job.error = "error", f"{type(e).__name__}: {e}"
                traceback.print_exc()
            finally:
                job.finished = time.time()

    @app.post("/api/jobs")
    def start_job(req: JobRequest):
        clip_path(req.clip)
        with queue_lock:
            for j in jobs.values():
                if j.clip == req.clip and j.status in ("queued", "running"):
                    return j.to_json()                   # already on its way
            job = Job(uuid.uuid4().hex[:12], req.clip, req.stages, req.backend or backend)
            jobs[job.id] = job
        threading.Thread(target=work, args=(job, req), daemon=True).start()
        return job.to_json()

    @app.get("/api/jobs")
    def list_jobs():
        return [j.to_json() for j in sorted(jobs.values(), key=lambda j: -j.created)]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "no such job")
        return jobs[job_id].to_json()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "no such job")
        cancel.add(job_id)
        return jobs[job_id].to_json()

    # ------------------------------------------------------------------ results
    @app.get("/api/clips/{clip}/report")
    def get_report(clip: str):
        return JSONResponse(load_report(clip))

    @app.get("/api/clips/{clip}/files/{rel:path}")
    def get_file(clip: str, rel: str):
        p = _safe_join(out / clip, rel)
        if not p.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(p, headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/clips/{clip}/video")
    def get_video(clip: str):
        return FileResponse(clip_path(clip))

    @app.get("/api/clips/{clip}/frame")
    def get_frame(clip: str, t: float = Query(0.0, ge=0), width: int = Query(480, ge=16, le=1920)):
        from PIL import Image

        from cinescope.io.video import VideoReader

        with VideoReader(clip_path(clip), width=width) as vr:
            f = next(iter(vr.frames(start_s=t)), None)
        if f is None:
            raise HTTPException(404, "past the end of the clip")
        buf = io.BytesIO()
        Image.fromarray(f.rgb).save(buf, "JPEG", quality=85)
        return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    def encoder(spec: str):
        if encoder_factory is not None:
            return encoder_factory(spec)
        from cinescope.pipeline.accel import clip_encoder
        from cinescope.pipeline.core import Options

        return clip_encoder(spec, Options(backend=backend))

    def index(clip: str):
        from cinescope.search.shotindex import ShotIndex

        rep = load_report(clip)
        if "search" not in rep:
            raise HTTPException(404, f"{clip!r} has no search index (run the 'search' stage)")
        return ShotIndex.load(out / clip / rep["search"]["index"]), rep["search"].get("spec", "openclip")

    def hit_json(clip: str, rel: str, h: dict) -> dict:
        frames = [f"/api/clips/{clip}/files/{rel}/{f['file']}" for f in h.get("frames", [])]
        return {"rank": h["rank"], "score": h["score"], "shot": h["shot"], "part": h.get("part", 0),
                "t0": h["t0"], "t1": h["t1"], "frames": frames,
                "tags": [t["tag"] for t in h.get("tags", [])][:5],
                "subtitles": [s["text"] for s in h.get("subtitles", [])],
                "characters": [c["name"] for c in h.get("characters", [])]}

    @app.get("/api/clips/{clip}/search")
    def search(clip: str, q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=50)):
        ix, spec = index(clip)
        enc = encoder(spec)
        rel = load_report(clip)["search"]["index"]
        t0 = time.perf_counter()
        hits = ix.search_text(q, enc, k=k)
        return {"query": q, "ms": round((time.perf_counter() - t0) * 1000, 1),
                "hits": [hit_json(clip, rel, h) for h in hits]}

    @app.get("/api/clips/{clip}/ask")
    def ask(clip: str, q: str = Query(..., min_length=1)):
        from cinescope.search.rag import answer

        ix, spec = index(clip)
        return answer(q, ix, encoder(spec), use_llm=False).to_json()

    @app.get("/api/clips/{clip}/campath")
    def campath(clip: str):
        rep = load_report(clip)
        if "camera_path" not in rep:
            raise HTTPException(404, "no camera path for this clip")
        return JSONResponse(json.loads((out / clip / rep["camera_path"]["file"]).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ the built front end
    if (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:
        @app.get("/")
        def no_frontend():
            return {"message": "CineScope API. Build the front end (cd app/frontend && npm ci && npm run build) "
                               "or run its dev server (npm run dev); API docs at /docs."}

    app.state.jobs = jobs
    app.state.out = out
    return app


_APP: FastAPI | None = None


def __getattr__(name: str):
    """`app` is built on first access (uvicorn's "app.backend.main:app"), so importing
    this module for create_app (the tests) has no side effects."""
    global _APP
    if name == "app":
        if _APP is None:
            _APP = create_app()
        return _APP
    raise AttributeError(name)
