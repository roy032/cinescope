import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

// The clip in a <video>. Browsers play only some codecs (H.264 MP4 and WebM, not
// AVI/MPEG-4 Part 2, and some Chromium builds have no H.264): when the video
// cannot play, the player shows decoded frames from the back end instead, with
// a scrubber.
export default function Player({ clip, report, seek, onTime }) {
  const ref = useRef(null);
  const [fallback, setFallback] = useState(false);
  const [t, setT] = useState(0);
  const dur = report.info.duration_s;

  useEffect(() => {
    setFallback(false);
    setT(0);
  }, [clip]);

  useEffect(() => {
    if (!seek) return;
    setT(seek.t);
    if (ref.current && !fallback) {
      ref.current.currentTime = seek.t;
      ref.current.play?.().catch(() => {});
    }
  }, [seek, fallback]);

  const aspect = `${report.info.width} / ${report.info.height}`;
  if (fallback) {
    return (
      <div className="player">
        <img src={api.frame(clip, t, 960)} alt={`frame at ${t.toFixed(1)} s`} style={{ aspectRatio: aspect }} />
        <input
          type="range"
          min={0}
          max={dur}
          step={0.1}
          value={t}
          aria-label="time"
          onChange={(e) => {
            const v = Number(e.target.value);
            setT(v);
            onTime(v);
          }}
        />
        <div className="note">This browser cannot play the file; showing decoded frames.</div>
      </div>
    );
  }
  return (
    <div className="player">
      <video
        ref={ref}
        src={api.video(clip)}
        controls
        preload="metadata"
        style={{ aspectRatio: aspect }}
        onTimeUpdate={(e) => onTime(e.currentTarget.currentTime)}
        onError={() => setFallback(true)}
      />
    </div>
  );
}
