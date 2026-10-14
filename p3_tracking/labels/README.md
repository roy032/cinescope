# Phase 3 labels

Made by Claude, by eye, on sheets rendered by the scripts below. No output of
the trackers, embedders or speaker detectors under test was shown while
labelling.

## `<clip>.ids.json` — face identities (Weeks 9 and 10)

`python p3_tracking/label_sheets.py <clip> --every N` samples every N-th frame,
draws every face YuNet proposes at score >= 0.3 with a number, and tiles the
frames into contact sheets. Each proposed box was given an identity name
(the same name for the same person all through the clip), dropped if it was
not a face, or marked `ignore` (background extras, the back of a head, faces
too small or blurred to identify; a tracker is neither rewarded nor
penalised there). Unclear frames were re-rendered at full resolution.

| Clip | Sampled | Boxes | Identities |
|---|---|---|---|
| goldeneye | every 12th frame (0.5 s), 165 frames | 202 (17 ignore) | 8: boris, natalya, avatar (the on-screen cartoon face), 5 control-room technicians |
| head-pose-face-detection-female-and-male | every 24th frame (2 s), 72 frames | 144 | 2: woman, man (always left / right of the frame; checked on every sheet) |
| face-demographics-walking | every 12th frame (1 s), 61 frames | 41 | 7, named by clothing |

Limits, stated plainly: the boxes are YuNet's, so the labels say *who* each
detection is and which proposals are not faces; they do not measure YuNet's
own misses (none were seen in the sampled frames except faces a few pixels
tall). Tracking scores are therefore about association, not detection.

## `goldeneye.speech.json` — who speaks when (Week 11)

`python p3_tracking/label_speech.py trailers/goldeneye.mp4` renders, for every
track, one row per second: the lower half of the aligned face at 12 fps, and
a bar where Silero VAD hears speech (a labeller cannot hear a picture, so
Silero stands in for the ears). Rule: a face is speaking where speech is
audible **and** its mouth visibly articulates. Intervals are per identity, so
they do not depend on the tracker. 12 `unsure` intervals (mouth hidden by a
hand or in profile, two plausible speakers, or Boris's final yell that
Silero does not hear under the music) are left out of scoring.

Caveats: lip reading by eye at 12 fps is coarse (±0.25 s at interval
edges), and because Silero decides audibility, the "ours with Silero VAD"
row in the results is favoured by construction. head-pose-face-detection-
female-and-male and face-demographics-walking have silent audio tracks
(RMS 0), so goldeneye is the only labelled dialogue clip.
