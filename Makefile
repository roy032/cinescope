# Thin wrapper over tools/run_everything.py (Windows: RUN_EVERYTHING.bat).
#   make all | make test | make pipeline CLIP=trailers/any_clip.mp4 | make kaggle
PYTHON ?= python3
CLIP ?= trailers/goldeneye.mp4
STEPS := setup fetch lint test frontend p2 pipeline accel docker kaggle

.PHONY: all $(STEPS)

all:
	$(PYTHON) tools/run_everything.py --clip $(CLIP)

$(STEPS):
	$(PYTHON) tools/run_everything.py $@ --clip $(CLIP)
