"""Slice a video timeline into fixed-length, possibly overlapping clip windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Window:
    """A window ``[start_sec, end_sec)`` with its position in the video's window list."""

    index: int
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


def compute_windows(
    duration_seconds: float,
    window_seconds: float,
    stride_seconds: float,
    min_window_seconds: float = 0.0,
) -> List[Window]:
    """Return sliding windows covering ``[0, duration_seconds)``.

    Full-length windows are placed every ``stride_seconds``. If the tail of the video
    is not covered, one final window ending at ``duration_seconds`` is added, provided it
    is at least ``min_window_seconds`` long (it is shortened if the video is shorter
    than one window).

    Example:
        ``compute_windows(10, 4, 2)`` -> [0-4], [2-6], [4-8], [6-10].
    """
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window_seconds and stride_seconds must be positive")
    if duration_seconds <= 0:
        return []
    eps = 1e-9
    windows: List[Window] = []
    start = 0.0
    while start + window_seconds <= duration_seconds + eps:
        windows.append(Window(len(windows), round(start, 6), round(start + window_seconds, 6)))
        start += stride_seconds
    covered = windows[-1].end_sec if windows else 0.0
    if covered < duration_seconds - eps:
        tail_start = max(0.0, duration_seconds - window_seconds)
        if duration_seconds - tail_start >= min_window_seconds - eps:
            windows.append(Window(len(windows), round(tail_start, 6), round(duration_seconds, 6)))
    return windows
