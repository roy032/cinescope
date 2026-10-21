import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import { useJob, useKeys } from "./hooks.js";
import { clamp, fmtSpeed, fmtTime, omitKeys, parseLocation, shotAt } from "./util.js";
import ClipBar from "./components/ClipBar.jsx";
import Player from "./components/Player.jsx";
import Barcode from "./components/Barcode.jsx";
import ShotStrip from "./components/ShotStrip.jsx";
import ShotStats, { Fact } from "./components/ShotStats.jsx";
import CharacterTimeline from "./components/CharacterTimeline.jsx";
import Subtitles from "./components/Subtitles.jsx";
import SearchPanel from "./components/SearchPanel.jsx";
import CameraPath from "./components/CameraPath.jsx";
import Timing from "./components/Timing.jsx";
import Exports from "./components/Exports.jsx";
import Timelines from "./components/Timelines.jsx";
import Cutout from "./components/Cutout.jsx";
import Look from "./components/Look.jsx";
import Editing from "./components/Editing.jsx";
import JobPanel from "./components/JobPanel.jsx";
import Shortcuts from "./components/Shortcuts.jsx";

const START = parseLocation(window.location.search);
const PANELLED = ["look", "editing"];

export default function App() {
  const [clips, setClips] = useState(null); // null until the first answer
  const [clip, setClip] = useState(START.clip);
  const [report, setReport] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | ready | missing | error
  const [error, setError] = useState("");
  const [time, setTime] = useState(START.t);
  const [startAt, setStartAt] = useState(START.t);
  const [uploading, setUploading] = useState(null); // 0..1 while a file uploads
  const [dragging, setDragging] = useState(false);
  const [help, setHelp] = useState(false);
  const [toast, setToast] = useState("");
  const player = useRef(null);
  const searchBox = useRef(null);

  const toastTimer = useRef(0);
  const say = useCallback((msg) => {
    setToast(msg);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(""), 2600);
  }, []);

  const refreshClips = useCallback(async () => {
    try {
      const cs = await api.clips();
      setClips(cs);
      setClip((cur) => (cur && cs.some((c) => c.clip === cur) ? cur : (cs.find((c) => c.processed) || cs[0])?.clip || ""));
    } catch (e) {
      setClips((c) => c || []);
      setError(`Cannot reach the CineScope server: ${e.message}`);
    }
  }, []);

  useEffect(() => {
    refreshClips();
  }, [refreshClips]);

  const loadReport = useCallback(async (name, quiet = false) => {
    if (!name) return;
    if (!quiet) {
      setStatus("loading");
      setReport(null);
    }
    try {
      setReport(await api.report(name));
      setStatus("ready");
    } catch (e) {
      if (e.status === 404) {
        setReport(null);
        setStatus("missing");
      } else {
        setStatus("error");
        setError(e.message);
      }
    }
  }, []);

  useEffect(() => {
    setError("");
    loadReport(clip);
  }, [clip, loadReport]);

  // the address bar always holds a link to this clip at this moment
  const second = Math.floor(time);
  useEffect(() => {
    if (!clip) return;
    const u = new URL(window.location.href);
    u.searchParams.set("clip", clip);
    if (second > 0) u.searchParams.set("t", String(second));
    else u.searchParams.delete("t");
    window.history.replaceState(null, "", u);
  }, [clip, second]);

  const job = useJob(clip, async () => {
    await refreshClips();
    await loadReport(clip, status === "ready");
    job.dismiss();
    say("Analysis finished");
  });

  const chooseClip = (name) => {
    if (name === clip) return;
    setStartAt(0);
    setTime(0);
    setClip(name);
  };

  const upload = async (file) => {
    setUploading(0);
    try {
      const r = await api.upload(file, setUploading);
      await refreshClips();
      chooseClip(r.clip);
      say(`Uploaded ${r.file}. Press Analyse to process it.`);
    } catch (e) {
      setError(`Upload failed: ${e.message}`);
    } finally {
      setUploading(null);
    }
  };

  // drop a video anywhere on the page to upload it
  useEffect(() => {
    let depth = 0;
    const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
    const enter = (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth += 1;
      setDragging(true);
    };
    const over = (e) => hasFiles(e) && e.preventDefault();
    const leave = () => {
      depth = Math.max(0, depth - 1);
      if (!depth) setDragging(false);
    };
    const drop = (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      setDragging(false);
      const f = e.dataTransfer.files?.[0];
      if (f) upload(f);
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragover", over);
    window.addEventListener("dragleave", leave);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragover", over);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("drop", drop);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const duration = report?.info?.duration_s || 0;
  const seekTo = useCallback(
    (t, play = false) => {
      const x = clamp(t, 0, Math.max(duration - 0.05, 0));
      setTime(x);
      player.current?.seek(x, play);
    },
    [duration],
  );
  const shots = report?.shots || [];
  const stepShot = (d) => {
    if (!shots.length) return;
    const now = player.current?.time?.() ?? time;
    let i = shotAt(shots, now);
    // "previous" from inside a shot first goes back to its start, like a media player
    if (d < 0 && i >= 0 && now - shots[i].t0 > 1) d = 0;
    i = clamp((i < 0 ? 0 : i) + d, 0, shots.length - 1);
    seekTo(shots[i].t0 + 0.01);
  };

  const share = async () => {
    const u = new URL(window.location.href);
    u.searchParams.set("t", time.toFixed(1));
    try {
      await navigator.clipboard.writeText(u.toString());
      say(`Link to ${fmtTime(time)} copied`);
    } catch {
      say("Copy the link from the address bar");
    }
  };

  useKeys({
    " ": () => player.current?.toggle(),
    k: () => player.current?.toggle(),
    ArrowLeft: () => seekTo(time - 5),
    ArrowRight: () => seekTo(time + 5),
    j: () => seekTo(time - 10),
    l: () => seekTo(time + 10),
    ",": () => stepShot(-1),
    ".": () => stepShot(1),
    "/": () => searchBox.current?.focus(),
    "?": () => setHelp((h) => !h),
    Escape: () => setHelp(false),
  });

  const ready = status === "ready" && report;
  // stages with a panel of their own are left out of "More timelines"
  const otherTimelines = omitKeys(report?.timelines, PANELLED);
  const look = report?.extra?.look?.summary;
  const editing = report?.extra?.editing;
  const sections = ready
    ? [
        report.shots && ["shots", "Shots"],
        editing && ["editing", "Editing"],
        look && ["look", "Look"],
        report.characters && ["characters", "Characters"],
        otherTimelines && ["timelines", "Timelines"],
        report.subtitles && ["subtitles", "Subtitles"],
        report.camera_path && ["camera", "Camera path"],
        report.shots && ["export", "Export"],
        report.timing && ["timing", "Timing"],
      ].filter(Boolean)
    : [];

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <span className="logo" aria-hidden="true" />
          <span className="name">CineScope</span>
          <span className="tag">an X-ray of any film, built from pixels up</span>
        </div>
        <ClipBar
          clips={clips || []}
          clip={clip}
          onClip={chooseClip}
          job={job}
          onStart={job.start}
          onUpload={upload}
          uploading={uploading}
        />
      </header>

      {sections.length > 1 && (
        <nav className="sections" aria-label="sections">
          {sections.map(([id, name]) => (
            <a key={id} href={`#${id}`}>
              {name}
            </a>
          ))}
          <span className="spacer" />
          <button className="link" onClick={() => setHelp(true)} title="keyboard shortcuts (?)">
            Shortcuts
          </button>
        </nav>
      )}

      {error && (
        <div className="error" role="alert">
          <span>{error}</span>
          <button className="link" onClick={() => setError("")} aria-label="dismiss">
            ✕
          </button>
        </div>
      )}

      {job.job && (status === "ready" || !["queued", "running"].includes(job.job.status)) && (
        <JobPanel job={job.job} onCancel={job.cancel} onDismiss={job.dismiss} />
      )}

      {clips && !clips.length && <Welcome onPick={() => document.querySelector(".clipbar input[type=file]")?.click()} />}

      {status === "loading" && <Skeleton />}

      {status === "missing" && clip && (
        <div className="panel empty">
          {job.active ? (
            <JobPanel job={job.job} onCancel={job.cancel} />
          ) : (
            <>
              <h2>{clip} has not been analysed yet</h2>
              <p className="dim">
                Analysing finds the cuts, colours and camera moves, who is on screen and talking, on-screen text, and builds a
                search index. On a CPU, expect roughly a few minutes per minute of video; finished stages are cached.
              </p>
              <div className="row-inline center">
                <button className="primary big" onClick={() => job.start(null)}>
                  Analyse {clip}
                </button>
                <button className="ghost" onClick={() => job.start(["shots", "palette", "camera"])} title="shots, palette and camera only: seconds, no models">
                  Quick look (cuts and colour only)
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {ready && (
        <main>
          <section className="row hero">
            <div className="player-col">
              <Player ref={player} clip={clip} report={report} startAt={startAt} onTime={setTime} />
              <div className="now">
                <span className="tnum">{fmtTime(time)}</span>
                <span className="dim tnum"> / {fmtTime(duration)}</span>
                {shots.length > 0 && shotAt(shots, time) >= 0 && <span className="dim"> · shot #{shots[shotAt(shots, time)].index}</span>}
                <span className="spacer" />
                <button className="link" onClick={share} title="copy a link that opens this clip at this moment">
                  Copy link to {fmtTime(time)}
                </button>
              </div>
              <Cutout clip={clip} time={time} />
            </div>
            <div className="side-col">
              <Summary report={report} />
              <SearchPanel key={clip} ref={searchBox} clip={clip} report={report} onSeek={seekTo} />
            </div>
          </section>

          {report.barcode && (
            <section className="panel">
              <h2>
                Movie barcode <small>the mean colour of every second; hover to read the time, click to jump</small>
              </h2>
              <Barcode
                colors={report.barcode.colors}
                duration={duration}
                time={time}
                onSeek={seekTo}
                describe={(t) => {
                  const i = shotAt(shots, t);
                  return i >= 0 ? `shot #${shots[i].index}` : null;
                }}
              />
            </section>
          )}

          {report.shots && (
            <section className="panel" id="shots">
              <h2>
                Shots{" "}
                <small>
                  {report.shots.length} shots · framing{report.framing_method ? ` (${report.framing_method})` : ""} · camera move ·
                  palette · press , and . to step through them
                </small>
              </h2>
              <ShotStats shots={report.shots} pacing={report.pacing} duration={duration} time={time} onSeek={seekTo} />
              <ShotStrip clip={clip} shots={report.shots} time={time} onSeek={seekTo} />
            </section>
          )}

          {editing && (
            <section className="panel" id="editing">
              <h2>
                Editing <small>pacing, camera set-ups and the patterns the cuts make</small>
              </h2>
              <Editing editing={editing} duration={duration} time={time} onSeek={seekTo} />
            </section>
          )}

          {look && (
            <section className="panel" id="look">
              <h2>
                Look <small>how it is lit and graded, shot by shot; hover to read a shot, click to jump</small>
              </h2>
              <Look shots={shots} summary={look} duration={duration} time={time} onSeek={seekTo} />
            </section>
          )}

          {report.characters && (
            <section className="panel" id="characters">
              <h2>
                Characters{" "}
                <small>
                  on screen and talking ({report.characters.asd} speaker detection, {report.characters.embedder} faces)
                </small>
              </h2>
              <CharacterTimeline clip={clip} chars={report.characters.list} duration={duration} time={time} onSeek={seekTo} />
            </section>
          )}

          {otherTimelines && (
            <section className="panel" id="timelines">
              <h2>
                More timelines <small>scene types, cut-outs and other stages</small>
              </h2>
              <Timelines timelines={otherTimelines} duration={duration} time={time} onSeek={seekTo} />
            </section>
          )}

          {(report.subtitles || report.camera_path) && (
            <section className="row two">
              {report.subtitles && (
                <div className="panel" id="subtitles">
                  <h2>
                    Subtitles <small>read from the picture ({report.subtitles.reader})</small>
                  </h2>
                  <Subtitles subs={report.subtitles} time={time} onSeek={seekTo} />
                </div>
              )}
              {report.camera_path && (
                <div className="panel" id="camera">
                  <h2>
                    Camera path <small>shot {report.camera_path.shot}, {report.camera_path.mode}; drag to turn</small>
                  </h2>
                  <CameraPath clip={clip} info={report.camera_path} time={time} onSeek={seekTo} />
                </div>
              )}
            </section>
          )}

          {report.shots && (
            <section className="panel" id="export">
              <h2>
                Export <small>hand the cut to an editor: every format cuts the original file at the same frames</small>
              </h2>
              <Exports clip={clip} report={report} />
            </section>
          )}

          {report.timing && (
            <section className="panel" id="timing">
              <h2>Pipeline timing</h2>
              <Timing report={report} />
            </section>
          )}
        </main>
      )}

      <footer>
        CineScope · results computed locally by cinescope.pipeline ·{" "}
        <button className="link" onClick={() => setHelp(true)}>
          keyboard shortcuts (?)
        </button>
      </footer>

      {help && <Shortcuts onClose={() => setHelp(false)} />}
      {dragging && (
        <div className="dropzone" aria-hidden="true">
          <div>Drop a video to upload it</div>
        </div>
      )}
      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
    </div>
  );
}

function Summary({ report }) {
  const i = report.info;
  const t = report.timing;
  const talk = report.characters?.list?.reduce((a, c) => a + c.speaking_time_s, 0) || 0;
  return (
    <div className="panel summary">
      <h1 title={report.video}>{report.clip}</h1>
      <div className="facts">
        <Fact k="length" v={fmtTime(i.duration_s)} />
        <Fact k="frame" v={`${i.width}×${i.height}`} />
        <Fact k="fps" v={i.fps.toFixed(2)} />
        {report.shots && <Fact k="shots" v={report.shots.length} />}
        {report.characters && <Fact k="characters" v={report.characters.list.length} />}
        {report.characters && <Fact k="talking" v={`${talk.toFixed(0)} s`} />}
        {report.subtitles && <Fact k="subtitle lines" v={report.subtitles.cues.length} />}
        {t && <Fact k="analysed at" v={`${fmtSpeed(t.video_s_per_wall_s)} speed`} title={`${t.wall_s.toFixed(1)} s for ${i.duration_s.toFixed(1)} s of video`} />}
      </div>
      {report.stages && <div className="dim small ran">Stages run: {report.stages.join(", ")}</div>}
    </div>
  );
}

function Welcome({ onPick }) {
  return (
    <div className="panel welcome">
      <h2>Welcome to CineScope</h2>
      <p>Give it any video and it finds every cut, the colour palette, camera moves, who is on screen and talking, on-screen text, and lets you search scenes in plain English.</p>
      <ol>
        <li>
          <b>Drop a video</b> anywhere on this page, or{" "}
          <button className="link" onClick={onPick}>
            choose a file
          </button>
          . MP4 (H.264) plays best in the browser; other formats still work.
        </li>
        <li>
          Or put videos in <code>trailers/</code> or <code>data/clips/</code> and reload.
        </li>
        <li>
          Press <b>Analyse</b>.
        </li>
      </ol>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="row hero skeleton" aria-busy="true" aria-label="loading">
      <div className="sk sk-video" />
      <div className="side-col">
        <div className="sk sk-box" />
        <div className="sk sk-box" />
      </div>
    </div>
  );
}
