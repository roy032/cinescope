"""SMPTE timecode and the frame rates editing software expects.

Video files report a float frame rate (23.976..., 29.97..., 25.0). Editing
software wants the nominal rate behind it: 24000/1001, 30000/1001, 25/1. The
timebase of a timecode is the rounded rate (24, 30, 25), and for 29.97 and
59.94 the drop-frame count skips frame numbers 0 and 1 (0-3 at 59.94) at the
start of every minute except each tenth, so that the timecode keeps up with
the wall clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

_NTSC = (24, 30, 48, 60, 120)


@dataclass(frozen=True)
class Rate:
    """A frame rate as num/den frames per second (24000/1001 for 23.976)."""

    num: int
    den: int

    @classmethod
    def from_fps(cls, fps: float) -> Rate:
        """The standard rate a measured fps stands for; anything non-standard is kept to 1/100 fps."""
        fps = float(fps)
        if fps <= 0:
            raise ValueError(f"frame rate must be positive, not {fps}")
        for n in _NTSC:
            if abs(fps - n * 1000 / 1001) < 0.005:
                return cls(n * 1000, 1001)
        if abs(fps - round(fps)) < 0.005:
            return cls(int(round(fps)), 1)
        f = Fraction(round(fps * 100), 100)
        return cls(f.numerator, f.denominator)

    @property
    def fps(self) -> float:
        return self.num / self.den

    @property
    def timebase(self) -> int:
        """The frames-per-second a timecode counts in (24 for 23.976)."""
        return max(1, round(self.num / self.den))

    @property
    def drop_frame_possible(self) -> bool:
        return self.den == 1001 and self.timebase in (30, 60)

    def frames(self, t: float) -> int:
        """The frame that shows at time t (seconds)."""
        return int(round(t * self.num / self.den))

    def seconds(self, frames: int) -> Fraction:
        return Fraction(frames * self.den, self.num)

    def rational(self, frames: int) -> str:
        """FCPXML's time format: frames * frame duration, as an exact fraction of a second."""
        f = self.seconds(frames)
        return f"{f.numerator}s" if f.denominator == 1 else f"{f.numerator}/{f.denominator}s"

    @property
    def frame_duration(self) -> str:
        return f"{self.den}/{self.num}s" if self.num % self.den else f"1/{self.num // self.den}s"


def to_timecode(frames: int, rate: Rate, drop_frame: bool = False) -> str:
    """Frame count -> "HH:MM:SS:FF" (non-drop) or "HH:MM:SS;FF" (drop-frame)."""
    if frames < 0:
        raise ValueError("timecode of a negative frame count")
    tb = rate.timebase
    if drop_frame:
        if not rate.drop_frame_possible:
            raise ValueError(f"drop-frame timecode needs 29.97 or 59.94 fps, not {rate.fps:.3f}")
        drop = 2 * tb // 30                       # 2 frame numbers at 29.97, 4 at 59.94
        per_10min = tb * 600 - 9 * drop
        per_min = tb * 60 - drop
        d, m = divmod(frames, per_10min)
        frames += 9 * drop * d + (drop * ((m - drop) // per_min) if m > drop else 0)
    ff = frames % tb
    s = frames // tb
    sep = ";" if drop_frame else ":"
    return f"{s // 3600 % 24:02d}:{s // 60 % 60:02d}:{s % 60:02d}{sep}{ff:02d}"


def from_timecode(tc: str, rate: Rate) -> int:
    """"HH:MM:SS:FF" or "HH:MM:SS;FF" -> frame count (the inverse of to_timecode)."""
    drop_frame = ";" in tc
    h, m, s, f = (int(x) for x in tc.replace(";", ":").split(":"))
    tb = rate.timebase
    frames = ((h * 60 + m) * 60 + s) * tb + f
    if drop_frame:
        drop = 2 * tb // 30
        minutes = h * 60 + m
        frames -= drop * (minutes - minutes // 10)
    return frames
