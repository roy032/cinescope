# extends: p2_train_log
# CenterNet v3: log sizes, crops that never cut a face 64 px or taller, and 30% "night" samples.
# Targets the two weaknesses the first checkpoints showed: Easy (large faces) scoring below Hard,
# and the dark movie frames (p2_detection/README.md, week 8).
TAG, SIZE_MODE, EPOCHS = "centernet_r18_v3", "log", 60
EXTRA = ["--keep-large", "64", "--low-light", "0.3"]
