import { useEffect, useRef } from "react";

const KEYS = [
  ["Space or K", "play / pause"],
  ["← / →", "back / forward 5 s"],
  ["J / L", "back / forward 10 s"],
  [", / .", "previous / next shot"],
  ["/", "search scenes"],
  ["?", "this help"],
  ["Esc", "close, or leave a text box"],
];

// The keyboard shortcuts, as a dialog.
export default function Shortcuts({ onClose }) {
  const ok = useRef(null);
  useEffect(() => ok.current?.focus(), []);
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal panel" role="dialog" aria-modal="true" aria-label="keyboard shortcuts" onClick={(e) => e.stopPropagation()}>
        <h2>Keyboard shortcuts</h2>
        <table className="keys">
          <tbody>
            {KEYS.map(([k, what]) => (
              <tr key={k}>
                <td>
                  {k.split(" / ").map((x, i) => (
                    <span key={x}>
                      {i > 0 && " / "}
                      <kbd>{x}</kbd>
                    </span>
                  ))}
                </td>
                <td>{what}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="dim small">Timelines, the barcode and the cutting rhythm also take the arrow keys when focused. Click anywhere on them to jump.</p>
        <button ref={ok} className="primary" onClick={onClose}>
          Got it
        </button>
      </div>
    </div>
  );
}
