import { useEffect, useRef } from "react";
import Scrub from "./Scrub.jsx";

// The movie barcode drawn from its colours (one per second), with a playhead.
export default function Barcode({ colors, duration, time, onSeek, describe }) {
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

  return (
    <Scrub className="barcode" duration={duration} time={time} onSeek={onSeek} describe={describe} label="movie barcode: click or use the arrow keys to seek">
      <canvas ref={ref} />
    </Scrub>
  );
}
