# Phase 7 labels

| File | What | How it was made |
|---|---|---|
| `sherlock_lines.txt` | 400 subtitle lines for the **test** set | Dialogue (text in quotes) from *The Adventures of Sherlock Holmes* (Conan Doyle, public domain, Project Gutenberg #1661), split at sentence ends, 12–72 characters, normalised to the recogniser's alphabet, shuffled with seed 7 |
| `tune_lines.txt` | 150 lines for the **tuning** set | The same, from the last 10% of *Pride and Prejudice* (the part the recogniser's training never sees) |
| `burned/<clip>.<style>.srt` | Truth for the test videos | `make_testset.py` on goldeneye, testvideo, Megamind (clips whose frames were never training backgrounds) — timings from reading speed, text burned in by FFmpeg/libass |
| `tune/<clip>.<style>.srt` | Truth for the tuning videos | `make_testset.py --lines tune_lines.txt` on people-detection and classroom (training-background clips) |
| `credits_test.json`, `credits_tune.json` | Lines of the synthetic credits clips | `make_credits_clip.py` (test: Megamind backgrounds, seed 3; tune: people-detection, seed 11) |
| `goldeneye_text.json` | Real on-screen text in the GoldenEye trailer | **By eye**, from full-resolution frames rendered with the Read tool, before any reader was run on the clip (details inside the file) |

The burned-in videos themselves are not committed (`outputs/p7/`); rerun
`make_testset.py` to rebuild them from the clips — the `.srt` files are
deterministic given the lines file and the clip's duration.
