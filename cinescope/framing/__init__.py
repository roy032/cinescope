"""Phase 1, week 5: shot-framing (shot scale) classification with a ResNet-18
written from scratch in PyTorch.

Scale classes follow MovieShots (Rao et al., ECCV 2020):
ECS extreme close-up, CS close-up, MS medium, FS full, LS long.
"""
SCALES = ("ECS", "CS", "MS", "FS", "LS")
SCALE_NAMES = {"ECS": "extreme close-up", "CS": "close-up", "MS": "medium shot",
               "FS": "full shot", "LS": "long shot"}
