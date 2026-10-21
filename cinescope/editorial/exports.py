"""A report's cuts as files editing software opens.

    edl(report)      CMX 3600 EDL: one event per shot (Premiere, Resolve, Avid, Final Cut via XML)
    fcpxml(report)   FCPXML 1.10: the clip cut at every shot, framing / camera / characters as
                     keywords, scene-type changes as chapter markers (Final Cut, Resolve, Premiere)
    otio(report)     OpenTimelineIO JSON (Resolve, Premiere, Avid, Nuke Studio through OTIO's adapters)
    shot_csv(report) the shot list a script supervisor or AD works from, one row per shot
    webvtt(report)   the subtitles read off the picture as WebVTT (the browser's <track>)

Every export cuts the original file at the times the report found, so an editor
can relink the source and start from CineScope's breakdown instead of a single
uncut clip. Each shot's in and out frame come from its times at the clip's
nominal rate, and every shot's out point is the next shot's in point, so the
cuts tile the clip with no gaps or overlaps.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape, quoteattr

from .timecode import Rate, from_timecode, to_timecode


def record_start(rate: Rate, drop_frame: bool) -> int:
    """The record frame of a sequence's first frame: 01:00:00:00, the editing convention."""
    return from_timecode("01:00:00;00" if drop_frame else "01:00:00:00", rate)


def rate_of(report: dict) -> Rate:
    return Rate.from_fps(report["info"]["fps"])


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def cut_list(report: dict) -> list[dict]:
    """The report's shots on the clip's frame grid, with the labels every export uses.

    Each row: index, f0, f1 (in and out frame, out exclusive), t0, t1, framing,
    camera, scene, characters (ids on screen for at least a third of the shot,
    or of their own time in it if shorter), thumb.
    """
    rate = rate_of(report)
    shots = report.get("shots") or []
    chars = (report.get("characters") or {}).get("list", [])
    rows = []
    for i, s in enumerate(shots):
        f0 = rate.frames(s["t0"])
        f1 = rate.frames(shots[i + 1]["t0"]) if i + 1 < len(shots) else max(rate.frames(s["t1"]), f0 + 1)
        f1 = max(f1, f0 + 1)
        fr = s.get("framing") or {}
        cam = s.get("camera") or {}
        scene = s.get("scenes") or {}
        dur = max(s["t1"] - s["t0"], 1e-6)
        on = []
        for c in chars:
            ov = sum(_overlap(s["t0"], s["t1"], a, b) for a, b in c.get("on_screen", []))
            if ov >= min(dur / 3, 1.0):
                on.append(c.get("name") or c["id"])
        rows.append({
            "index": s["index"], "f0": f0, "f1": f1, "t0": s["t0"], "t1": s["t1"],
            "framing": fr.get("label"), "framing_name": fr.get("name"),
            "camera": cam.get("label") if cam.get("label") not in (None, "unknown") else None,
            "scene": scene.get("type") if isinstance(scene, dict) else None,
            "characters": on, "thumb": s.get("thumb"),
        })
    return rows


def _describe(row: dict) -> str:
    parts = [row["framing_name"] or row["framing"], row["camera"], row["scene"],
             ", ".join(row["characters"]) if row["characters"] else None]
    return " | ".join(p for p in parts if p)


def _drop_frame(rate: Rate, drop_frame: bool | None) -> bool:
    return rate.drop_frame_possible if drop_frame is None else drop_frame


# ----------------------------------------------------------------------------- EDL
def edl(report: dict, title: str | None = None, reel: str = "AX", drop_frame: bool | None = None) -> str:
    """CMX 3600. Source timecode is the file's own time from 00:00:00:00; the
    record side starts at 01:00:00:00. Reel "AX" plus a FROM CLIP NAME comment
    is how Premiere and Resolve relink an event to a file instead of a tape."""
    rate = rate_of(report)
    df = _drop_frame(rate, drop_frame)
    tc = lambda f: to_timecode(f, rate, df)  # noqa: E731
    rec0 = record_start(rate, df)
    name = report.get("video") or report.get("clip", "clip")
    title = (title or f"{report.get('clip', 'clip')} CineScope cuts")[:70]
    lines = [f"TITLE: {title}", f"FCM: {'DROP FRAME' if df else 'NON-DROP FRAME'}", ""]
    for n, r in enumerate(cut_list(report), start=1):
        lines.append(f"{n:03d}  {reel[:8]:<8} V     C        {tc(r['f0'])} {tc(r['f1'])} "
                     f"{tc(rec0 + r['f0'])} {tc(rec0 + r['f1'])}")
        lines.append(f"* FROM CLIP NAME: {name}")
        d = _describe(r)
        if d:
            lines.append(f"* COMMENT: SHOT {r['index']} | {d}")
        lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------------------------- FCPXML
def _file_url(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if len(path) > 1 and path[1] == ":":                     # a Windows path, as recorded by the report
        return "file:///" + quote(path.replace("\\", "/"), safe="/:")
    return p.as_uri() if p.is_absolute() else quote(path)


def fcpxml(report: dict, event: str = "CineScope") -> str:
    """FCPXML 1.10: one asset (the source file) and a project whose spine is that
    asset cut into the report's shots. Keywords carry framing, camera move and
    the characters on screen, so Final Cut's keyword collections group shots by
    them; a chapter marker starts each new scene type."""
    rate = rate_of(report)
    info = report["info"]
    rows = cut_list(report)
    total = rows[-1]["f1"] if rows else max(rate.frames(info["duration_s"]), 1)
    asset_dur = max(total, rate.frames(info["duration_s"]))
    clip = report.get("clip", "clip")
    tcf = "DF" if rate.drop_frame_possible else "NDF"
    r = rate.rational
    out = ['<?xml version="1.0" encoding="UTF-8"?>', "<!DOCTYPE fcpxml>", "", '<fcpxml version="1.10">',
           "  <resources>",
           f'    <format id="r1" frameDuration="{rate.frame_duration}" width="{info["width"]}" '
           f'height="{info["height"]}"/>',
           f'    <asset id="r2" name={quoteattr(clip)} start="0s" duration="{r(asset_dur)}" hasVideo="1" '
           f'hasAudio="1" format="r1" videoSources="1" audioSources="1" audioChannels="2">',
           f'      <media-rep kind="original-media" src={quoteattr(_file_url(report.get("video_path")))}/>',
           "    </asset>", "  </resources>", "  <library>", f"    <event name={quoteattr(event)}>",
           f"      <project name={quoteattr(clip + ' (CineScope cuts)')}>",
           f'        <sequence format="r1" duration="{r(total)}" tcStart="0s" tcFormat="{tcf}">',
           "          <spine>"]
    scene = None
    for row in rows:
        f0, n = row["f0"], row["f1"] - row["f0"]
        out.append(f'            <asset-clip ref="r2" name={quoteattr("Shot " + str(row["index"]))} offset="{r(f0)}" '
                   f'start="{r(f0)}" duration="{r(n)}" format="r1" tcFormat="{tcf}">')
        if row["scene"] and row["scene"] != scene:
            out.append(f'              <chapter-marker start="{r(f0)}" duration="{rate.frame_duration}" '
                       f'value={quoteattr(row["scene"])} posterOffset="0s"/>')
        scene = row["scene"] or scene
        kws = [k for k in (row["framing_name"] or row["framing"], row["camera"], row["scene"]) if k]
        kws += row["characters"]
        if kws:
            out.append(f'              <keyword start="{r(f0)}" duration="{r(n)}" '
                       f'value={quoteattr(", ".join(dict.fromkeys(kws)))}/>')
        out.append("            </asset-clip>")
    out += ["          </spine>", "        </sequence>", "      </project>", "    </event>", "  </library>",
            "</fcpxml>", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------- OTIO
def _rt(value: int, rate: float) -> dict:
    return {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": float(value)}


def _range(start: int, dur: int, rate: float) -> dict:
    return {"OTIO_SCHEMA": "TimeRange.1", "duration": _rt(dur, rate), "start_time": _rt(start, rate)}


def otio(report: dict) -> str:
    """An OpenTimelineIO timeline: one video track, a clip per shot, each with
    CineScope's labels in its metadata and a marker per scene-type change."""
    rate = rate_of(report)
    fps = rate.fps
    rows = cut_list(report)
    clip = report.get("clip", "clip")
    full = max(rate.frames(report["info"]["duration_s"]), rows[-1]["f1"] if rows else 1)
    ref = {"OTIO_SCHEMA": "ExternalReference.1", "metadata": {}, "name": report.get("video") or clip,
           "available_range": _range(0, full, fps), "target_url": _file_url(report.get("video_path"))}
    children, scene = [], None
    for row in rows:
        markers = []
        if row["scene"] and row["scene"] != scene:
            markers.append({"OTIO_SCHEMA": "Marker.2", "metadata": {}, "name": row["scene"], "color": "PURPLE",
                            "comment": "", "marked_range": _range(row["f0"], 0, fps)})
        scene = row["scene"] or scene
        meta = {k: row[k] for k in ("index", "framing", "framing_name", "camera", "scene", "characters")}
        children.append({"OTIO_SCHEMA": "Clip.1", "name": f"Shot {row['index']}", "effects": [],
                         "markers": markers, "media_reference": dict(ref), "metadata": {"cinescope": meta},
                         "source_range": _range(row["f0"], row["f1"] - row["f0"], fps)})
    track = {"OTIO_SCHEMA": "Track.1", "kind": "Video", "name": "V1", "children": children, "effects": [],
             "markers": [], "metadata": {}, "source_range": None}
    tl = {"OTIO_SCHEMA": "Timeline.1", "name": f"{clip} (CineScope cuts)",
          "global_start_time": _rt(record_start(rate, rate.drop_frame_possible), fps),
          "metadata": {"cinescope": {"generated": report.get("generated"), "stages": report.get("stages", [])}},
          "tracks": {"OTIO_SCHEMA": "Stack.1", "name": "tracks", "children": [track], "effects": [],
                     "markers": [], "metadata": {}, "source_range": None}}
    return json.dumps(tl, indent=2)


# ---------------------------------------------------------------------------- CSV
CSV_FIELDS = ["shot", "tc_in", "tc_out", "start_s", "end_s", "duration_s", "frames", "framing", "camera",
              "scene", "characters", "thumbnail"]


def shot_csv(report: dict) -> str:
    rate = rate_of(report)
    df = rate.drop_frame_possible
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_FIELDS)
    for r in cut_list(report):
        w.writerow([r["index"], to_timecode(r["f0"], rate, df), to_timecode(r["f1"], rate, df),
                    f"{r['t0']:.3f}", f"{r['t1']:.3f}", f"{r['t1'] - r['t0']:.3f}", r["f1"] - r["f0"],
                    r["framing_name"] or r["framing"] or "", r["camera"] or "", r["scene"] or "",
                    "; ".join(r["characters"]), r["thumb"] or ""])
    return buf.getvalue()


# --------------------------------------------------------------------------- WebVTT
def _vtt_time(t: float) -> str:
    ms = int(round(max(t, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def webvtt(report: dict) -> str:
    """The subtitle cues as WebVTT. A cue's text is escaped (WebVTT is markup)."""
    cues = (report.get("subtitles") or {}).get("cues", [])
    out = ["WEBVTT", ""]
    for i, c in enumerate(sorted(cues, key=lambda c: c["t0"]), start=1):
        t1 = max(c["t1"], c["t0"] + 0.04)
        text = "\n".join(escape(line) for line in str(c["text"]).replace("-->", "->").splitlines() if line.strip())
        if not text:
            continue
        out += [str(i), f"{_vtt_time(c['t0'])} --> {_vtt_time(t1)}", text, ""]
    return "\n".join(out)


#: format name -> (function, file extension, media type)
FORMATS = {
    "edl": (edl, "edl", "text/plain; charset=utf-8"),
    "fcpxml": (fcpxml, "fcpxml", "application/xml"),
    "otio": (otio, "otio", "application/json"),
    "csv": (shot_csv, "csv", "text/csv; charset=utf-8"),
    "vtt": (webvtt, "vtt", "text/vtt; charset=utf-8"),
}
