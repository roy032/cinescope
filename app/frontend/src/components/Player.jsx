import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { api } from "../api.js";
import { clamp, fmtTime } from "../util.js";

// The clip in a <video>. Browsers play only some codecs (H.264 MP4 and WebM, not
// AVI/MPEG-4 Part 2, and some Chromium builds have no H.264): when the video
// cannot play, the player shows frames decoded by the back end instead, with its
// own play button and scrubber, at whatever rate the back end decodes them.
//
// The parent drives it through a ref: toggle() plays or pauses, seek(t, play).
const Player = forwardRef(function Player({ clip, report, startAt = 0, onTime }, ref) {
  const video = useRef(null);
  const [fallback, setFallback] = useState(false);
  const [t, setT] = useState(startAt);
  const [playing, setPlaying] = useState(false);
  const [shown, setShown] = useState("");
  const shownAt = useRef(0);
  const dur = report.info.duration_s;

  useEffect(() => {
    setFallback(false);
    setPlaying(false);
    setT(startAt);
    // startAt is read once per clip, on purpose
  }, [clip]); // eslint-disable-line react-hooks/exhaustive-deps

  const go = (v, play) => {
    const x = clamp(v, 0, Math.max(dur - 0.05, 0));
    setT(x);
    onTime(x);
    if (!fallback && video.current) {
      video.current.currentTime = x;
      if (play) video.current.play?.().catch(() => {});
    }
  };

  useImperativeHandle(ref, () => ({
    toggle() {
      if (fallback) play(!playing);
      else if (video.current) {
        if (video.current.paused) video.current.play().catch(() => {});
        else video.current.pause();
      }
    },
    seek: go,
    time: () => (fallback || !video.current ? t : video.current.currentTime),
  }));

  // the fallback's "playback": after each frame arrives, ask for the frame as far
  // ahead as the wall clock has moved (so it keeps real time, dropping frames)
  const frameLoaded = (e) => {
    setShown(e.currentTarget.getAttribute("src"));
    if (!playing) return;
    const now = performance.now();
    const step = Math.max(0.2, (now - shownAt.current) / 1000);
    shownAt.current = now;
    if (t + step >= dur) {
      setPlaying(false);
      return;
    }
    setT(t + step);
    onTime(t + step);
  };
  const play = (on) => {
    setPlaying(on);
    if (on) {
      shownAt.current = performance.now();
      const nv = t >= dur - 0.25 ? 0 : t + 0.2; // a new frame starts the loop; at the end, start over
      setT(nv);
      onTime(nv);
    }
  };

  const aspect = `${report.info.width} / ${report.info.height}`;
  if (fallback) {
    const src = api.frame(clip, t, 960);
    return (
      <div className="player fallback">
        <div className="screen" style={{ aspectRatio: aspect }}>
          <img src={src} alt={`frame at ${fmtTime(t)}`} onLoad={frameLoaded} onError={() => play(false)} />
          {shown !== src && !playing && <span className="spinner" aria-hidden="true" />}
        </div>
        <div className="fb-controls">
          <button className="icon" onClick={() => play(!playing)} aria-label={playing ? "pause" : "play"}>
            {playing ? "❚❚" : "▶"}
          </button>
          <input
            type="range"
            min={0}
            max={dur}
            step={0.04}
            value={t}
            aria-label="time"
            onChange={(e) => {
              setPlaying(false);
              go(Number(e.target.value), false);
            }}
          />
          <span className="tnum small">
            {fmtTime(t)} <span className="dim">/ {fmtTime(dur)}</span>
          </span>
        </div>
        <div className="note">This browser cannot play this file, so the app shows frames decoded on the server (no sound).</div>
      </div>
    );
  }
  return (
    <div className="player">
      <video
        ref={video}
        src={api.video(clip)}
        controls
        preload="metadata"
        style={{ aspectRatio: aspect }}
        onLoadedMetadata={(e) => {
          if (startAt > 0) e.currentTarget.currentTime = startAt;
        }}
        onTimeUpdate={(e) => onTime(e.currentTarget.currentTime)}
        onSeeked={(e) => onTime(e.currentTarget.currentTime)}
        onError={() => setFallback(true)}
      />
    </div>
  );
});

export default Player;
