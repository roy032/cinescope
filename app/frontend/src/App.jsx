import { useCallback, useEffect, useRef, useState } from "react";
import { api, fmtTime } from "./api.js";
import ClipBar from "./components/ClipBar.jsx";
import Player from "./components/Player.jsx";
import Barcode from "./components/Barcode.jsx";
import ShotStrip from "./components/ShotStrip.jsx";
import CharacterTimeline from "./components/CharacterTimeline.jsx";
import Subtitles from "./components/Subtitles.jsx";
import SearchPanel from "./components/SearchPanel.jsx";
import CameraPath from "./components/CameraPath.jsx";
import Timing from "./components/Timing.jsx";
import Timelines from "./components/Timelines.jsx";
import Cutout from "./components/Cutout.jsx";

function initialClip() {
  const p = new URLSearchParams(window.location.search);
  return p.get("clip") || "";
}

export default function App() {
  const [clips, setClips] = useState([]);
  const [clip, setClip] = useState(initialClip());
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  const [time, setTime] = useState(0);
  const [seek, setSeek] = useState(null); // {t, n}: a request for the player
  const seekN = useRef(0);

  const refreshClips = useCallback(async () => {
    try {
      const cs = await api.clips();
      setClips(cs);
      if (!clip && cs.length) setClip((cs.find((c) => c.processed) || cs[0]).clip);
    } catch (e) {
      setError(String(e));
    }
  }, [clip]);

  useEffect(() => {
    refreshClips();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const loadReport = useCallback(async (name) => {
    setReport(null);
    setError("");
    if (!name) return;
    try {
      setReport(await api.report(name));
    } catch {
      setReport(null); // not processed yet: the clip bar offers to run it
    }
  }, []);

  useEffect(() => {
    loadReport(clip);
    setTime(0);
    if (clip) {
      const u = new URL(window.location.href);
      u.searchParams.set("clip", clip);
      window.history.replaceState(null, "", u);
    }
  }, [clip, loadReport]);

  const seekTo = useCallback((t) => {
    seekN.current += 1;
    setSeek({ t, n: seekN.current });
    setTime(t);
  }, []);

  const duration = report?.info?.duration_s || 0;

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <span className="logo" aria-hidden="true" />
          <span className="name">CineScope</span>
          <span className="tag">an X-ray of any film, built from pixels up</span>
        </div>
        <ClipBar
          clips={clips}
          clip={clip}
          onClip={setClip}
          onDone={async () => {
            await refreshClips();
            await loadReport(clip);
          }}
          onUploaded={async (name) => {
            await refreshClips();
            setClip(name);
          }}
        />
      </header>

      {error && <div className="error">{error}</div>}

      {!report && clip && (
        <div className="empty">
          <p>
            <b>{clip}</b> has not been analysed yet. Press <b>Analyse</b> to run the pipeline on it.
          </p>
        </div>
      )}

      {report && (
        <main>
          <section className="row hero">
            <div className="player-col">
              <Player clip={clip} report={report} seek={seek} onTime={setTime} />
              <div className="now">
                <span>{fmtTime(time)}</span>
                <span className="dim"> / {fmtTime(duration)}</span>
              </div>
              <Cutout clip={clip} time={time} />
            </div>
            <div className="side-col">
              <Summary report={report} />
              <SearchPanel clip={clip} report={report} onSeek={seekTo} />
            </div>
          </section>

          {report.barcode && (
            <section className="panel">
              <h2>Movie barcode <small>one line per second, mean colour</small></h2>
              <Barcode colors={report.barcode.colors} duration={duration} time={time} onSeek={seekTo} />
            </section>
          )}

          {report.shots && (
            <section className="panel">
              <h2>
                Shots <small>{report.shots.length} shots · framing{report.framing_method ? ` (${report.framing_method})` : ""} · camera move · palette</small>
              </h2>
              <ShotStrip clip={clip} shots={report.shots} time={time} onSeek={seekTo} />
            </section>
          )}

          {report.characters && (
            <section className="panel">
              <h2>
                Characters <small>on screen and talking ({report.characters.asd} speaker detection, {report.characters.embedder} faces)</small>
              </h2>
              <CharacterTimeline clip={clip} chars={report.characters.list} duration={duration} time={time} onSeek={seekTo} />
            </section>
          )}

          {report.timelines && (
            <section className="panel">
              <h2>More timelines <small>from stages added later</small></h2>
              <Timelines timelines={report.timelines} duration={duration} time={time} onSeek={seekTo} />
            </section>
          )}

          <section className="row two">
            {report.subtitles && (
              <div className="panel">
                <h2>Subtitles <small>read from the picture ({report.subtitles.reader})</small></h2>
                <Subtitles subs={report.subtitles} time={time} onSeek={seekTo} />
              </div>
            )}
            {report.camera_path && (
              <div className="panel">
                <h2>Camera path <small>shot {report.camera_path.shot}, {report.camera_path.mode}; drag to turn</small></h2>
                <CameraPath clip={clip} info={report.camera_path} time={time} onSeek={seekTo} />
              </div>
            )}
          </section>

          <section className="panel">
            <h2>Pipeline timing</h2>
            <Timing report={report} />
          </section>
        </main>
      )}
      <footer>CineScope · Phase 9 · results computed locally by cinescope.pipeline</footer>
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
        {report.subtitles && <Fact k="subtitles" v={report.subtitles.cues.length} />}
        {t && <Fact k="processed at" v={`${t.video_s_per_wall_s}× real time`} />}
      </div>
    </div>
  );
}

function Fact({ k, v }) {
  return (
    <div className="fact">
      <div className="v">{v}</div>
      <div className="k">{k}</div>
    </div>
  );
}
