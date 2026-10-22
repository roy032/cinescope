import { api } from "../api.js";

// Hand the cut to an editor: the shots CineScope found as files editing software
// opens (each cuts the original video at the same frames, so the shot numbers match
// across all of them), the subtitles as WebVTT, a printable storyboard, and a
// one-file HTML breakdown to send to anyone.
const FORMATS = [
  ["html", "Breakdown", "one shareable page: the facts, charts and every shot", "opens in any browser, offline"],
  ["edl", "EDL", "CMX 3600, one event per shot", "opens in Premiere, Resolve, Avid"],
  ["fcpxml", "FCPXML", "the clip cut at every shot, labels as keywords", "opens in Final Cut Pro, Resolve, Premiere"],
  ["otio", "OpenTimelineIO", "the cut with each shot's labels in its metadata", "opens in Resolve, Premiere, Avid, Nuke"],
  ["csv", "Shot list", "timecodes, length, framing, camera, characters", "opens in Excel or Google Sheets"],
  ["pdf", "Storyboard", "every shot's frame with its timecode and labels", "to print and mark up"],
  ["vtt", "Subtitles", "the subtitles read off the picture", "for players, YouTube and editors"],
  ["mp4", "Highlights reel", "the strongest moments cut together, with sound", "plays anywhere; rendered on first download"],
  ["highlights", "Highlights EDL", "the reel's moments as a CMX 3600 EDL", "rebuild and refine it in an editor"],
];

const EXT = { highlights: "edl" };

export default function Exports({ clip, report }) {
  const has = (fmt) => {
    if (fmt === "vtt") return report.subtitles?.cues?.length > 0;
    if (fmt === "mp4" || fmt === "highlights") return report.extra?.highlights?.moments?.length > 0;
    return report.shots?.length > 0;
  };
  return (
    <div className="exports">
      {FORMATS.filter(([fmt]) => has(fmt)).map(([fmt, name, what, opens]) => (
        <a key={fmt} className="export" href={api.exportUrl(clip, fmt)} download>
          <span className="ext">.{EXT[fmt] || fmt}</span>
          <span className="en">{name}</span>
          <span className="dim small">{what}</span>
          <span className="dim small">{opens}</span>
        </a>
      ))}
    </div>
  );
}
