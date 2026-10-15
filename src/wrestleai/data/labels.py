"""Map labeled time ranges ``(start_sec, end_sec, label)`` onto clip windows."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from wrestleai.data.windowing import Window


@dataclass(frozen=True)
class LabelRange:
    """A labeled span of a video."""

    start_sec: float
    end_sec: float
    label: str
    notes: Optional[str] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.end_sec <= self.start_sec:
            raise ValueError(f"Label range end ({self.end_sec}) must be after start ({self.start_sec})")


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the intersection of two intervals (0 if disjoint)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_window_label(
    window: Window,
    ranges: Iterable[LabelRange],
    default_label: str,
    overlap_threshold: float = 0.5,
) -> str:
    """Label a window by the range label that covers the largest fraction of it.

    Coverage is summed per label (so two adjacent "highlight" ranges count together).
    The best label wins if its coverage fraction is >= ``overlap_threshold``; otherwise
    the window gets ``default_label``.
    """
    if window.duration <= 0:
        return default_label
    coverage: Dict[str, float] = {}
    for r in ranges:
        if r.label == default_label:
            continue
        ov = overlap(window.start_sec, window.end_sec, r.start_sec, r.end_sec)
        if ov > 0:
            coverage[r.label] = coverage.get(r.label, 0.0) + ov
    if not coverage:
        return default_label
    best_label, best_cov = max(coverage.items(), key=lambda kv: kv[1])
    fraction = min(1.0, best_cov / window.duration)
    return best_label if fraction >= overlap_threshold - 1e-9 else default_label


def label_windows(
    windows: Sequence[Window],
    ranges: Sequence[LabelRange],
    default_label: str,
    overlap_threshold: float = 0.5,
) -> List[str]:
    """Vectorised :func:`assign_window_label` over a list of windows."""
    return [assign_window_label(w, ranges, default_label, overlap_threshold) for w in windows]


def load_label_file(path: str | Path) -> List[LabelRange]:
    """Read ranges from CSV (``start_sec,end_sec,label[,notes]`` header) or JSON (list of objects)."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
    else:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    return [LabelRange(float(r["start_sec"]), float(r["end_sec"]), str(r["label"]), r.get("notes") or None)
            for r in rows]


def save_label_file(ranges: Sequence[LabelRange], path: str | Path) -> None:
    """Write ranges as CSV with a ``start_sec,end_sec,label,notes`` header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["start_sec", "end_sec", "label", "notes"])
        for r in ranges:
            writer.writerow([f"{r.start_sec:.3f}", f"{r.end_sec:.3f}", r.label, r.notes or ""])
