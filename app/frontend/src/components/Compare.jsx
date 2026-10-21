import { useState } from "react";
import { api } from "../api.js";

// Set this clip's style against any other processed clip: opens the one-file
// comparison page (similarity, biggest differences, curves, every metric).
export default function Compare({ clip, clips }) {
  const others = clips.filter((c) => c.processed && c.clip !== clip);
  const [other, setOther] = useState("");
  const pick = others.some((c) => c.clip === other) ? other : others[0]?.clip || "";
  if (!others.length) return <p className="dim">Process another clip to compare it with this one.</p>;
  return (
    <div className="compare">
      <label>
        Compare <b>{clip}</b> with{" "}
        <select value={pick} onChange={(e) => setOther(e.target.value)}>
          {others.map((c) => (
            <option key={c.clip} value={c.clip}>
              {c.clip}
            </option>
          ))}
        </select>
      </label>
      <a className="export-btn" href={api.compareUrl(clip, pick)} target="_blank" rel="noreferrer">
        Open the comparison
      </a>
      <span className="dim small">style similarity, the biggest differences, pace and intensity side by side</span>
    </div>
  );
}
