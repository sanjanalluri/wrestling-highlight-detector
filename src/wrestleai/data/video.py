"""OpenCV video ingestion: metadata probing, frame extraction and FPS conversion.

Ported from the original Groot ``video_analyzer.py`` and ``frame_rate_converter.py``.
Frame selection is timestamp-based rather than ``frame_idx % interval``, so it handles
both down- and up-sampling and non-integer FPS ratios (e.g. 29.97 -> 12).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    """Basic properties of a video file."""

    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def _open(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    return cap


def probe_video(path: str | Path) -> VideoInfo:
    """Read FPS, frame count and resolution from a video file."""
    cap = _open(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if fps <= 0:
        raise IOError(f"Video reports invalid FPS ({fps}): {path}")
    return VideoInfo(str(path), fps, frame_count, width, height)


def find_videos(directory: str | Path, extensions: Iterable[str]) -> List[Path]:
    """Recursively list video files under ``directory`` with the given extensions."""
    exts = {e.lower() for e in extensions}
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Video directory not found: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _resize_rgb(frame_bgr: np.ndarray, resize: Optional[Tuple[int, int]]) -> np.ndarray:
    if resize is not None:
        h, w = resize
        frame_bgr = cv2.resize(frame_bgr, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def extract_frames(
    path: str | Path,
    fps: float,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    resize: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode frames from ``[start_sec, end_sec)`` resampled to ``fps``.

    Args:
        path: video file.
        fps: output sampling rate.
        start_sec: start of the segment.
        end_sec: end of the segment (``None`` = end of video).
        resize: optional ``(height, width)``.

    Returns:
        ``(frames, timestamps)`` where ``frames`` is ``uint8`` RGB with shape
        ``(T, H, W, 3)`` and ``timestamps`` holds the source time of each frame.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    cap = _open(path)
    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0:
            raise IOError(f"Video reports invalid FPS: {path}")
        start_idx = max(0, int(round(start_sec * src_fps)))
        if start_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames: List[np.ndarray] = []
        stamps: List[float] = []
        step = 1.0 / fps
        next_t = start_sec
        idx = start_idx
        half_frame = 0.5 / src_fps
        while True:
            t = idx / src_fps
            if end_sec is not None and t >= end_sec:
                break
            ok, frame = cap.read()
            if not ok:
                break
            # Emit this source frame for every output timestamp it is closest to
            # (repeats frames when upsampling).
            while next_t < t + half_frame and (end_sec is None or next_t < end_sec):
                frames.append(_resize_rgb(frame, resize))
                stamps.append(t)
                next_t += step
            idx += 1
    finally:
        cap.release()
    if not frames:
        h, w = resize if resize else (0, 0)
        return np.zeros((0, h, w, 3), dtype=np.uint8), np.zeros((0,), dtype=np.float64)
    return np.stack(frames), np.asarray(stamps, dtype=np.float64)


def sample_uniform(frames: np.ndarray, n: int) -> np.ndarray:
    """Pick ``n`` frames evenly spaced over ``frames`` (repeating when there are fewer)."""
    if len(frames) == 0:
        raise ValueError("Cannot sample from an empty frame sequence")
    idx = np.linspace(0, len(frames) - 1, num=n).round().astype(int)
    return frames[idx]


def read_window(
    path: str | Path,
    start_sec: float,
    end_sec: float,
    fps: float,
    num_frames: int,
    resize: Tuple[int, int],
) -> np.ndarray:
    """Return exactly ``num_frames`` RGB frames ``(num_frames, H, W, 3)`` for one window."""
    frames, _ = extract_frames(path, fps, start_sec, end_sec, resize)
    if len(frames) == 0:
        # Seeking past the last decodable frame: fall back to the window start.
        frames, _ = extract_frames(path, fps, max(0.0, start_sec - 1.0 / fps), None, resize)
        if len(frames) == 0:
            raise IOError(f"No frames decoded from {path} in [{start_sec}, {end_sec})")
        frames = frames[:1]
    return sample_uniform(frames, num_frames)


def write_video(path: str | Path, frames_rgb: Sequence[np.ndarray], fps: float) -> None:
    """Write RGB frames to an ``.mp4`` (mp4v codec, available in all OpenCV builds)."""
    if len(frames_rgb) == 0:
        raise ValueError("No frames to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise IOError(f"Could not create video writer for {path}")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def convert_video(
    src: str | Path,
    dst: str | Path,
    target_fps: float,
    resize: Optional[Tuple[int, int]] = None,
) -> VideoInfo:
    """Re-encode ``src`` at ``target_fps`` (optionally resized) without loading it all into RAM.

    Equivalent of the original Groot frame-rate converter; a smaller, lower-FPS copy makes
    repeated window decoding during training much faster.
    """
    cap = _open(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[cv2.VideoWriter] = None
    written = 0
    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0:
            raise IOError(f"Video reports invalid FPS: {src}")
        step, next_t, idx, half = 1.0 / target_fps, 0.0, 0, 0.5 / src_fps
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / src_fps
            while next_t < t + half:
                if resize is not None:
                    frame_out = cv2.resize(frame, (resize[1], resize[0]), interpolation=cv2.INTER_AREA)
                else:
                    frame_out = frame
                if writer is None:
                    h, w = frame_out.shape[:2]
                    writer = cv2.VideoWriter(
                        str(dst), cv2.VideoWriter_fourcc(*"mp4v"), float(target_fps), (w, h)
                    )
                    if not writer.isOpened():
                        raise IOError(f"Could not create video writer for {dst}")
                writer.write(frame_out)
                written += 1
                next_t += step
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if written == 0:
        raise IOError(f"No frames decoded from {src}")
    logger.info("Converted %s -> %s (%d frames @ %.2f fps)", src, dst, written, target_fps)
    return probe_video(dst)
