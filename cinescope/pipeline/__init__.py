"""Phase 9: the whole of CineScope on one clip, in one command.

    python -m cinescope.pipeline trailers/goldeneye.mp4
    python -m cinescope.pipeline clip.mp4 --stages shots palette camera --backend onnx --workers 2

    from cinescope.pipeline import run
    report = run("clip.mp4", stages=["shots", "characters"])

See core.py for the stage interface, stages.py for the built-in stages,
accel.py for the ONNX / TensorRT backends and report.py for report.json.
"""
from .core import STAGES, Context, Interrupted, Options, Stage, Stream, register, run

__all__ = ["STAGES", "Context", "Interrupted", "Options", "Stage", "Stream", "register", "run"]
