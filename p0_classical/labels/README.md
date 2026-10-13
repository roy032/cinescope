Shot-boundary labels (`<clip>.labels.json`), one per clip, made with
`p0_classical/label_cuts.py` or by reading every frame. These are the Phase 0
test set: the detector's parameters were never tuned on these clips. Add a label
file for any clip and `eval_cuts.py --all` scores it too.

Format:

```json
{"video": "your_clip.mp4", "fps": 23.976, "n_frames": 7218, "labeller": "human",
 "transitions": [{"kind": "cut", "frame": 131, "t": 5.464},
                 {"kind": "dissolve", "start": 880, "end": 902, "t": 37.12}]}
```

`frame` is the first frame of the new shot (0-based, in decoding order).
