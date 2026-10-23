import { useEffect, useState } from "react";
import { api } from "../api.js";
import { readyDefaults } from "../util.js";

// What this machine can run (GET /api/doctor): a line that says how many stages are
// ready, the fix for each one that is not, and a button to analyse with the ready ones.
export default function Setup({ onStart }) {
  const [d, setD] = useState(null);
  useEffect(() => {
    let live = true;
    api.doctor().then((x) => live && setD(x), () => {});
    return () => {
      live = false;
    };
  }, []);
  if (!d || !d.not_ready?.length) return null;
  const notReady = d.not_ready.filter((n) => d.stages[n]?.default !== false);
  if (!notReady.length) return null;
  const ready = readyDefaults(d);
  return (
    <details className="setup">
      <summary>
        {notReady.length} of {Object.keys(d.stages).length} stages cannot run on this machine yet:{" "}
        {notReady.join(", ")}. <span className="link">Show how to fix it</span>
      </summary>
      <ul>
        {notReady.map((n) => (
          <li key={n}>
            <b>{n}</b>: {d.stages[n].missing.map((m) => m.fix).join("; ")}
          </li>
        ))}
      </ul>
      <p className="dim small">
        Analyse still runs every stage: the ones above are reported as failed and the rest finish. Or run only the{" "}
        {ready.length} ready ones:
      </p>
      <button className="ghost" onClick={() => onStart(ready)}>
        Analyse with the {ready.length} ready stages
      </button>
    </details>
  );
}
