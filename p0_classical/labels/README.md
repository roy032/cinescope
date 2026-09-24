Hand-made shot-boundary labels (`<clip>.labels.json`), produced with
`p0_classical/label_cuts.py`. These are the Phase 0 test set: the detector's
parameters were never tuned on these clips.

Format:

```json
{"video": "Ted_Lasso_900s.mkv", "fps": 23.976, "n_frames": 7218, "labeller": "human",
 "transitions": [{"kind": "cut", "frame": 131, "t": 5.464},
                 {"kind": "dissolve", "start": 880, "end": 902, "t": 37.12}]}
```

`frame` is the first frame of the new shot (0-based, in decoding order).
