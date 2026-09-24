# Box labels for the clip comparison

`<clip>_<tag>.boxes.json` files from `label_boxes.py` go here: three clips,
chosen to stress detectors in different ways — **dark** (low light), **profile**
(side-on and turned-away faces), **crowd** (many small, overlapping faces).
About 30 frames each is enough to separate two detectors; label every face,
marking the ones you cannot judge as *ignore* (`i`).

The frames themselves stay in `outputs/` (git-ignored); only the JSON is committed.
