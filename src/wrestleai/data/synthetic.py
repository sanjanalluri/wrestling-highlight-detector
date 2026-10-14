"""Tiny synthetic "match" videos with known highlight ranges, for demos and tests.

Each video shows two coloured blobs ("wrestlers") drifting slowly on a mat. During
highlight ranges they move quickly and a bright marker appears, so the label is
learnable from a handful of frames. This is test data only; it says nothing about
performance on real footage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from wrestleai.data.labels import LabelRange, save_label_file
from wrestleai.data.video import write_video

logger = logging.getLogger(__name__)


def _highlight_ranges(
    rng: np.random.Generator, duration: float, n: int, min_len: float, max_len: float
) -> List[LabelRange]:
    """Place ``n`` non-overlapping ranges with at least 1s of gap."""
    ranges: List[LabelRange] = []
    attempts = 0
    while len(ranges) < n and attempts < 1000:
        attempts += 1
        length = float(rng.uniform(min_len, max_len))
        start = float(rng.uniform(0.5, max(0.5, duration - length - 0.5)))
        end = start + length
        if all(end + 1.0 <= r.start_sec or start >= r.end_sec + 1.0 for r in ranges):
            ranges.append(LabelRange(round(start, 2), round(end, 2), "highlight"))
    return sorted(ranges, key=lambda r: r.start_sec)


def render_synthetic_video(
    duration: float,
    fps: float,
    size: Tuple[int, int],
    ranges: List[LabelRange],
    seed: int,
) -> np.ndarray:
    """Render RGB frames ``(T, H, W, 3)`` for the given highlight ranges."""
    rng = np.random.default_rng(seed)
    h, w = size
    n_frames = int(round(duration * fps))
    pos = np.array([[w * 0.35, h * 0.5], [w * 0.65, h * 0.5]], dtype=np.float64)
    frames = []
    for i in range(n_frames):
        t = i / fps
        active = any(r.start_sec <= t < r.end_sec for r in ranges)
        speed = 0.18 * min(h, w) if active else 0.01 * min(h, w)
        pos += rng.normal(0.0, speed, size=pos.shape)
        pos[:, 0] = np.clip(pos[:, 0], 0.15 * w, 0.85 * w)
        pos[:, 1] = np.clip(pos[:, 1], 0.15 * h, 0.85 * h)
        frame = np.full((h, w, 3), (70, 110, 150), dtype=np.uint8)  # blue-ish mat
        noise = rng.integers(-8, 9, size=(h, w, 1))
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        radius = max(2, int(0.1 * min(h, w)))
        cv2.circle(frame, tuple(int(v) for v in pos[0]), radius, (200, 40, 40), -1)
        cv2.circle(frame, tuple(int(v) for v in pos[1]), radius, (240, 240, 240), -1)
        if active:
            center = tuple(int(v) for v in pos.mean(axis=0))
            cv2.circle(frame, center, int(radius * 1.6), (255, 220, 0), max(1, radius // 3))
        frames.append(frame)
    return np.stack(frames)


def make_synthetic_dataset(
    out_dir: str | Path,
    num_videos: int = 3,
    duration: float = 24.0,
    fps: float = 12.0,
    size: Tuple[int, int] = (96, 96),
    highlights_per_video: int = 2,
    seed: int = 0,
) -> List[Tuple[Path, Path]]:
    """Write ``num_videos`` synthetic videos plus ``<name>.labels.csv`` files.

    Returns:
        ``(video_path, labels_path)`` pairs.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pairs: List[Tuple[Path, Path]] = []
    for v in range(num_videos):
        ranges = _highlight_ranges(rng, duration, highlights_per_video, 3.0, 5.0)
        frames = render_synthetic_video(duration, fps, size, ranges, seed=seed * 1000 + v)
        video_path = out / f"synthetic_match_{v:02d}.mp4"
        labels_path = out / f"synthetic_match_{v:02d}.labels.csv"
        write_video(video_path, list(frames), fps)
        save_label_file(ranges, labels_path)
        logger.info("Wrote %s (%d frames) with %d highlight ranges", video_path, len(frames), len(ranges))
        pairs.append((video_path, labels_path))
    return pairs
