import { useEffect, useRef } from "react";

// The movie barcode drawn from its colours (one per second), with a playhead.
export default function Barcode({ colors, duration, time, onSeek }) {
  const ref = useRef(null);

  useEffect(() => {
    const c = ref.current;
    if (!c || !colors?.length) return;
    c.width = colors.length;
    c.height = 1;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(colors.length, 1);
    colors.forEach(([r, g, b], i) => {
      img.data.set([r, g, b, 255], i * 4);
    });
    ctx.putImageData(img, 0, 0);
  }, [colors]);

  const click = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    onSeek(((e.clientX - r.left) / r.width) * duration);
  };

  return (
    <div className="barcode" onClick={click} role="button" tabIndex={0} aria-label="movie barcode, click to seek">
      <canvas ref={ref} />
      {duration > 0 && <div className="playhead" style={{ left: `${(100 * time) / duration}%` }} />}
    </div>
  );
}
