// Unit tests for the pure helpers: `npm test` (Node's built-in runner, no extra packages).
import assert from "node:assert/strict";
import { test } from "node:test";
import { eta, fmtDuration, fmtSpeed, fmtTime, jobSteps, pacePath, parseLocation, shotAt, shotStats } from "./util.js";

test("fmtTime: minutes, hours, rounding carry", () => {
  assert.equal(fmtTime(0), "0:00.0");
  assert.equal(fmtTime(75.34), "1:15.3");
  assert.equal(fmtTime(59.96), "1:00.0");
  assert.equal(fmtTime(7384.2), "2:03:04.2");
  assert.equal(fmtTime(-3), "0:00.0");
  assert.equal(fmtTime(undefined), "0:00.0");
});

test("fmtDuration and fmtSpeed", () => {
  assert.equal(fmtDuration(0.42), "0.4 s");
  assert.equal(fmtDuration(42.4), "42 s");
  assert.equal(fmtDuration(185), "3 min 5 s");
  assert.equal(fmtDuration(120), "2 min");
  assert.equal(fmtDuration(4800), "1 h 20 min");
  assert.equal(fmtSpeed(1.9338), "1.9×");
  assert.equal(fmtSpeed(71.926), "72×");
  assert.equal(fmtSpeed(0.042), "0.04×");
});

const shots = [
  { index: 0, t0: 0, t1: 2, duration_s: 2, camera: { label: "static" }, framing: { label: "CS", name: "close-up" } },
  { index: 1, t0: 2, t1: 3, duration_s: 1, camera: { label: "pan" }, framing: { label: "LS", name: "long shot" } },
  { index: 2, t0: 3, t1: 9, duration_s: 6, camera: { label: "unknown" }, framing: { label: "CS", name: "close-up" } },
];

test("shotAt finds the shot playing at t", () => {
  assert.equal(shotAt(shots, 0), 0);
  assert.equal(shotAt(shots, 2), 1);
  assert.equal(shotAt(shots, 8.9), 2);
  assert.equal(shotAt(shots, 9), -1);
  assert.equal(shotAt([], 1), -1);
});

test("shotStats: ASL, median, cuts per minute, screen-time shares", () => {
  const st = shotStats(shots, 9);
  assert.equal(st.n, 3);
  assert.equal(st.asl, 3);
  assert.equal(st.median, 2);
  assert.equal(st.longest.index, 2);
  assert.ok(Math.abs(st.cutsPerMin - (60 * 2) / 9) < 1e-9);
  assert.deepEqual(st.framing.map((r) => r.label), ["close-up", "long shot"]);
  assert.ok(Math.abs(st.framing[0].frac - 8 / 9) < 1e-9);
  assert.deepEqual(st.camera.map((r) => r.label), ["static", "pan"]); // "unknown" is left out
  assert.equal(shotStats([], 0), null);
});

test("jobSteps: done, running, cached and waiting steps", () => {
  const job = {
    status: "running",
    plan: ["decode", "shots", "palette", "camera"],
    stage_progress: { palette: 0.5 },
  };
  assert.deepEqual(
    jobSteps(job).map((s) => s.state),
    ["cached", "cached", "running", "waiting"],
  );
  const done = { ...job, status: "done", stage_progress: { decode: 1, shots: 1 } };
  assert.deepEqual(
    jobSteps(done).map((s) => s.state),
    ["done", "done", "cached", "cached"],
  );
  const stopped = { ...job, status: "cancelled", stage_progress: { decode: 0.3 } };
  assert.deepEqual(
    jobSteps(stopped).map((s) => s.state),
    ["stopped", "skipped", "skipped", "skipped"],
  );
  assert.deepEqual(jobSteps(null), []);
});

test("eta waits for enough progress", () => {
  assert.equal(eta(1, 0.5), null);
  assert.equal(eta(10, 0.01), null);
  assert.equal(eta(10, 0.5), 10);
  assert.equal(eta(10, 1), null);
});

test("parseLocation", () => {
  assert.deepEqual(parseLocation("?clip=goldeneye&t=12.5"), { clip: "goldeneye", t: 12.5 });
  assert.deepEqual(parseLocation("?t=abc"), { clip: "", t: 0 });
  assert.deepEqual(parseLocation(""), { clip: "", t: 0 });
});

test("pacePath: scaled to the box, fastest at the top, closed for a fill", () => {
  const curve = { t: [0, 5, 10], cuts_per_min: [0, 12, 6] };
  assert.equal(pacePath(curve, 10, 100, 42), "M 0.0 41.0 L 50.0 1.0 L 100.0 21.0");
  assert.ok(pacePath(curve, 10, 100, 42, true).endsWith("L 100.0 42 L 0.0 42 Z"));
  assert.equal(pacePath(null, 10, 100, 42), "");
  assert.equal(pacePath({ t: [0], cuts_per_min: [1] }, 10, 100, 42), "");
});
