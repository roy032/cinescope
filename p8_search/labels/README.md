# Scene-search queries (hand-written)

`queries.json` holds 50 natural-language queries for nine public clips, with the
ground truth for each.

How they were made: by eye, by Claude, from contact sheets rendered for this purpose
(three frames per detected shot for the multi-shot clips — goldeneye, testvideo,
Megamind — and one frame every 2.5 s for the single-shot clips), **before any search
system had been run on these clips**. No query was edited after seeing results.

- Multi-shot clips list the relevant Phase 0 shot ids (`cinescope/shots/pipeline.py`
  with default parameters) and their time spans; single-shot clips list time intervals
  in seconds read off the sheets (±1 s precision).
- Evaluation uses the times, so the labels survive a different segmentation: a
  retrieved segment is relevant when it overlaps a ground-truth interval by at least
  half of the shorter of the two (`cinescope/search/metrics.py`).
- `vtest` is the **dev** clip: the chat's "no match" threshold was set on it
  (`calibrate_chat.py`). Every other clip is **test** and was scored once.
- The trailers contain no explosion and no car chase; the roadmap's example queries
  are used only as demos of the "not found" answer.

Clips: download from PySceneDetect's test resources (goldeneye, testvideo), OpenCV's
samples (Megamind.avi, vtest.avi) and intel-iot-devkit/sample-videos (the people clips).
