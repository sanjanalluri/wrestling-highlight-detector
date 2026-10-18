"""Optical-flow motion descriptors, appended to the CNN features of each frame.

For every frame on the ``data.target_fps`` grid, dense Farneback flow is computed against
the previous grid frame on a small grayscale copy. Broadcast footage pans and zooms, so
the median flow vector (camera motion) is subtracted first; the remaining flow is what
the wrestlers are doing. It is summarised into :data:`MOTION_DIM` numbers:

* residual flow magnitude statistics (mean, std, p50, p90, p99, max; log-scaled)
* fraction of pixels moving more than 1 and 3 pixels
* camera motion (magnitude, dx, dy)
* a 4x4 grid of mean residual magnitude (where on screen the action is)
* an 8-bin direction histogram of residual flow, weighted by magnitude

Descriptors are cached per video like the CNN features (see :mod:`wrestleai.features.cache`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch

from wrestleai.config import Config
from wrestleai.data.dataset import WindowSample

logger = logging.getLogger(__name__)

GRID = 4
DIRECTION_BINS = 8
MOTION_DIM = 6 + 2 + 3 + GRID * GRID + DIRECTION_BINS  # 35


def motion_descriptor(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Return the :data:`MOTION_DIM`-dim float32 descriptor of motion from ``prev_gray`` to ``gray``."""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    camera = np.median(flow.reshape(-1, 2), axis=0)
    resid = flow - camera
    mag = np.linalg.norm(resid, axis=-1)
    stats = np.log1p([mag.mean(), mag.std(), *np.percentile(mag, [50, 90, 99]), mag.max()])
    moving = [(mag > 1.0).mean(), (mag > 3.0).mean()]
    cam = [np.log1p(np.linalg.norm(camera)), camera[0] / 10.0, camera[1] / 10.0]
    h, w = mag.shape
    grid = mag[: h - h % GRID, : w - w % GRID].reshape(GRID, h // GRID, GRID, w // GRID).mean(axis=(1, 3))
    angle = np.arctan2(resid[..., 1], resid[..., 0])
    hist, _ = np.histogram(angle, bins=DIRECTION_BINS, range=(-np.pi, np.pi), weights=mag)
    hist = hist / (hist.sum() + 1e-6)
    return np.concatenate([stats, moving, cam, np.log1p(grid).ravel(), hist]).astype(np.float32)


def motion_cache_key(cfg: Config) -> str:
    relevant = {"fps": cfg.data.target_fps, "size": list(cfg.features.motion_size), "version": 1}
    return "motion-" + hashlib.sha1(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]


class MotionCache:
    """Computes and stores per-frame motion descriptors for each video."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.dir = Path(cfg.paths.feature_cache_dir) / motion_cache_key(cfg)

    def _file(self, video_path: str) -> Path:
        p = Path(video_path).resolve()
        return self.dir / f"{p.stem}-{hashlib.sha1(str(p).encode()).hexdigest()[:8]}.npz"

    @staticmethod
    def _stamp(video_path: str) -> np.ndarray:
        st = Path(video_path).stat()
        return np.array([st.st_size, int(st.st_mtime)], dtype=np.int64)

    def _load(self, video_path: str) -> Dict[int, np.ndarray]:
        f = self._file(video_path)
        if not f.exists():
            return {}
        with np.load(f) as data:
            if not np.array_equal(data["stamp"], self._stamp(video_path)):
                return {}
            return {int(k): v for k, v in zip(data["ks"], data["feats"])}

    def _save(self, video_path: str, feats: Dict[int, np.ndarray]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        ks = np.array(sorted(feats), dtype=np.int32)
        np.savez(self._file(video_path), ks=ks, feats=np.stack([feats[int(k)] for k in ks]),
                 stamp=self._stamp(video_path))

    def _compute(self, video_path: str, needed: Sequence[int]) -> Dict[int, np.ndarray]:
        from wrestleai.features.cache import iter_grid_frames

        wanted, out = set(int(k) for k in needed), {}
        started, prev, last = time.time(), None, None
        for k, rgb in iter_grid_frames(video_path, self.cfg.data.target_fps, tuple(self.cfg.features.motion_size)):
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if k in wanted:
                last = motion_descriptor(prev if prev is not None else gray, gray)
                out[k] = last
            prev = gray
            if k >= max(wanted):
                break
        zero = np.zeros(MOTION_DIM, dtype=np.float32)
        for k in wanted - set(out):  # past the last decodable frame: no motion
            out[k] = zero
        logger.info("Motion features for %d frames of %s in %.0fs", len(wanted), Path(video_path).name,
                    time.time() - started)
        return out

    def window_features(self, samples: Sequence[WindowSample]) -> torch.Tensor:
        """Return float32 motion descriptors ``(N, frames_per_window, MOTION_DIM)``."""
        from wrestleai.features.cache import window_frame_indices

        fps, n = self.cfg.data.target_fps, self.cfg.data.frames_per_window
        indices = [window_frame_indices(s.start_sec, s.end_sec, fps, n) for s in samples]
        by_video: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            by_video[s.video_path].append(i)
        result: List[np.ndarray] = [np.empty(0)] * len(samples)
        for video, sample_ids in sorted(by_video.items()):
            needed = sorted({int(k) for i in sample_ids for k in indices[i]})
            feats = self._load(video)
            missing = [k for k in needed if k not in feats]
            if missing:
                feats.update(self._compute(video, missing))
                self._save(video, feats)
            for i in sample_ids:
                result[i] = np.stack([feats[int(k)] for k in indices[i]])
        return torch.from_numpy(np.stack(result).astype(np.float32))
