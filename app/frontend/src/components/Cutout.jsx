import { useState } from "react";
import { api, fmtTime } from "../api.js";

// Phase 4 on demand: cut the largest person out of the frame at the playhead.
export default function Cutout({ clip, time }) {
  const [method, setMethod] = useState("grabcut");
  const [req, setReq] = useState(null); // {url, t}
  const [state, setState] = useState("");
  const go = () => {
    setState("cutting…");
    setReq({ url: api.cutout(clip, time, method), t: time });
  };
  return (
    <div className="cutout">
      <div className="row-inline">
        <button className="ghost" onClick={go}>
          Cut out the actor at {fmtTime(time)}
        </button>
        <select value={method} onChange={(e) => setMethod(e.target.value)} aria-label="segmenter">
          <option value="grabcut">GrabCut (ours)</option>
          <option value="unet">U-Net (ours)</option>
          <option value="yolo">YOLOv8n-seg</option>
          <option value="mobilesam">MobileSAM</option>
        </select>
        {state && <span className="dim small">{state}</span>}
      </div>
      {req && (
        <img
          key={req.url}
          src={req.url}
          alt={`cut-out at ${req.t.toFixed(2)} s`}
          onLoad={() => setState("")}
          onError={() => setState("no cut-out (no person found, or the model is missing)")}
        />
      )}
    </div>
  );
}
