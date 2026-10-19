import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";

const ACTIVE = ["queued", "running"];

// The analysis job of one clip: picks up a job already running on the server (after a
// reload, or one started from another tab), polls it, and calls onDone when it finishes.
export function useJob(clip, onDone) {
  const [job, setJob] = useState(null);
  const done = useRef(onDone);
  done.current = onDone; // a ref, so a re-render (the player ticks 4x a second) never resets the poll

  useEffect(() => {
    setJob(null);
    if (!clip) return undefined;
    let live = true;
    api
      .jobs(clip)
      .then((js) => live && setJob(js.find((j) => ACTIVE.includes(j.status)) || null))
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [clip]);

  const id = job?.id;
  const active = !!job && ACTIVE.includes(job.status);
  useEffect(() => {
    if (!active) return undefined;
    let live = true;
    let timer;
    const tick = async () => {
      try {
        const j = await api.job(id);
        if (!live) return;
        setJob(j);
        if (ACTIVE.includes(j.status)) timer = setTimeout(tick, 1000);
        else if (j.status === "done") done.current?.(j);
      } catch {
        if (live) setJob((j) => j && { ...j, status: "error", error: "lost contact with the server (did it restart?)" });
      }
    };
    timer = setTimeout(tick, 600);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [id, active]);

  const start = useCallback(
    async (stages, backend) => {
      try {
        setJob(await api.startJob(clip, stages, backend));
      } catch (e) {
        setJob({ status: "error", error: e.message });
      }
    },
    [clip],
  );
  const cancel = useCallback(async () => {
    if (id) setJob(await api.cancel(id));
  }, [id]);
  const dismiss = useCallback(() => setJob(null), []);

  return { job, active, start, cancel, dismiss };
}

// Global keyboard shortcuts, ignored while typing in a field.
export function useKeys(map) {
  const ref = useRef(map);
  ref.current = map;
  useEffect(() => {
    const h = (e) => {
      const el = e.target;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (el.closest?.("input, textarea, select, [contenteditable=true]")) {
        if (e.key === "Escape") el.blur();
        return;
      }
      const f = ref.current[e.key];
      if (f) {
        // Space on a focused button should press it, not toggle playback
        if (e.key === " " && el.closest?.("button, [role=button], video")) return;
        e.preventDefault();
        f(e);
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);
}

// Close a popover on outside click or Escape.
export function useDismiss(open, setOpen, ref) {
  useEffect(() => {
    if (!open) return undefined;
    const click = (e) => ref.current && !ref.current.contains(e.target) && setOpen(false);
    const key = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", click);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mousedown", click);
      document.removeEventListener("keydown", key);
    };
  }, [open, setOpen, ref]);
}
