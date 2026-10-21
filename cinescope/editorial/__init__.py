"""Editorial hand-off: a report's cuts as EDL / FCPXML / OpenTimelineIO / CSV,
subtitles as WebVTT, a printable storyboard, and the editing pace.

    python -m cinescope.editorial outputs/<clip>/report.json            # every format -> outputs/<clip>/exports/
    python -m cinescope.editorial outputs/<clip>/report.json -f edl fcpxml -o somewhere/
"""
from .exports import FORMATS, cut_list, edl, fcpxml, otio, shot_csv, webvtt
from .pacing import pacing
from .timecode import Rate, from_timecode, to_timecode

__all__ = ["FORMATS", "Rate", "cut_list", "edl", "fcpxml", "from_timecode", "otio", "pacing", "shot_csv",
           "to_timecode", "webvtt"]
