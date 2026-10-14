"""Timestamp parsing/formatting shared by the labeler and inference output."""

from __future__ import annotations


def parse_time(text: str) -> float:
    """Parse ``SS``, ``SS.s``, ``MM:SS`` or ``HH:MM:SS`` into seconds.

    Unlike the original Groot labeler, ``0:00`` is a valid time (0.0 seconds).

    Raises:
        ValueError: if the string is not a valid non-negative timestamp.
    """
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3 or any(p == "" for p in parts):
        raise ValueError(f"Invalid time {text!r}; use SS, MM:SS or HH:MM:SS")
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid time {text!r}; use SS, MM:SS or HH:MM:SS") from exc
    if any(v < 0 for v in values) or any(v >= 60 for v in values[1:]):
        raise ValueError(f"Invalid time {text!r}; minutes/seconds fields must be in [0, 60)")
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return seconds


def format_time(seconds: float) -> str:
    """Format seconds as ``M:SS.s`` (or ``H:MM:SS.s`` past an hour)."""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours >= 1:
        return f"{int(hours)}:{int(minutes):02d}:{secs:04.1f}"
    return f"{int(minutes)}:{secs:04.1f}"
