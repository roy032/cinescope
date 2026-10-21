"""Film language: how a film is lit, graded, framed, cut and mixed.

These read what the pipeline already has (the shot list, the tiny thumbnail of
every frame, the soundtrack) and describe it in the words a cinematographer,
editor or sound mixer would use:

    look.py      exposure key, contrast, saturation, colour temperature,
                 teal-and-orange, symmetry, per shot and for the whole clip
    editing.py   pacing curve, recurring camera setups, shot/reverse-shot
                 exchanges and fast-cut montages
    sound.py     loudness (EBU R 128 style), peaks, silences, onsets, speech
    story.py     scenes, an intensity curve, three acts, turning points, climax
    composition.py
                 saliency, centre of interest, rule of thirds, balance, headroom,
                 shallow or deep focus

No models, no training: every number is a few lines of NumPy over pixels, and
every threshold is fixed up front (none are tuned on the public test clips).
"""
