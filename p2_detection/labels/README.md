# Box labels for the clip comparison

`<clip>_<tag>.boxes.json` files from `label_boxes.py` go here: three clips,
chosen to stress detectors in different ways — **dark** (low light), **profile**
(side-on and turned-away faces), **crowd** (many small, overlapping faces).
About 30 frames each is enough to separate two detectors; label every face,
marking the ones you cannot judge as *ignore* (`i`).

The frames themselves stay in `outputs/` (git-ignored); only the JSON is committed.

## The labelled public clips

| File | Clip | Frames | Faces (+ ignored) | Other classes |
|---|---|---|---|---|
| `goldeneye_dark.boxes.json` | `goldeneye.mp4`, 43–82 s every 1.33 s: low-key office interiors, a night exterior in snow, a face lit only by a match | 30 | 30 (+9) | person 36 (+11), tv 6 |
| `head-pose-face-detection-female-and-male_profile.boxes.json` | `head-pose-face-detection-female-and-male.mp4` (Intel sample video, 768×432 upscaled to 1280): head tilts, heads down, three-quarter views and full profiles | 30 | 57 (+3) | person 60 |
| `classroom_crowd.boxes.json` | `classroom.mp4` (Intel sample video), 0.5–32.5 s every 1.06 s: four people at desks, faces 51–90 px tall (median 68) in a 1280 × 720 frame | 31 | 124 | person 124 |

Regenerate the frames (the labels stay valid) with:

```bash
python p2_detection/label_boxes.py trailers/goldeneye.mp4 --start 43 --seconds 39.5 --fps 0.75 --tag dark --classes "face,person,tv,car"
python p2_detection/label_boxes.py trailers/head-pose-face-detection-female-and-male.mp4 --tag profile --classes "face,person" \
    --times 5,6.5,8,9.5,11,18,19.5,21,22.5,58,60,62,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140
python p2_detection/label_boxes.py trailers/classroom.mp4 --start 0.5 --seconds 32 --fps 0.94 --tag crowd --classes "face,person"
```

**How these were labelled.** By eye, without a detector anywhere in the loop:
every frame was rendered with a labelled pixel grid (and zoomed, brightened
crops for small or dark faces); box corners were read off the grid, then the
boxes were drawn back onto the frames to check them. Face boxes follow WIDER
FACE: top of the forehead to the chin, across the visible face (ear to nose
for a profile). Ignored: faces too blurred or dark to judge, faces hidden by
hair (heads bowed), a caricature on a screen, and people reduced to a hand or
the back of a head. Detectors were run only after all three files were
written, and no box was changed afterwards.

The "crowd" clip is a small crowd — four people — because no public clip
reachable from this machine has a real one; `vtest.avi` has more people, but
their faces are a few pixels tall and would all be ignored.
