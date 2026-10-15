"""On-disk cache of per-frame CNN features for frozen-backbone training.

When the feature extractor (and segmenter, if enabled) is frozen, its output for a given
frame never changes, so it only needs to be computed once. This module decodes each video
a single time at ``data.target_fps``, runs [segmentation ->] CNN on exactly the frames the
clip windows sample, and stores them per video. Training then only fits the Bi-LSTM and
classifier on ``(T, D)`` feature sequences, which takes minutes instead of hours on CPU.

Frame selection matches :func:`wrestleai.data.video.read_window`: frames on the global
``target_fps`` grid inside ``[start, end)``, then ``frames_per_window`` of them evenly spaced.
The cache is keyed by every setting that changes the features, plus the video's size and
modification time, so stale entries are never reused.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import cv2
import numpy as np
import torch

from wrestleai.config import Config
from wrestleai.data.dataset import WindowSample, frames_to_tensor

logger = logging.getLogger(__name__)

FEATURE_BATCH = 32


def uses_frozen_features(cfg: Config) -> bool:
    """True when per-frame features are fixed, so they can be cached."""
    seg_frozen = (not cfg.segmentation.enabled) or cfg.segmentation.freeze
    return cfg.training.cache_features and cfg.features.freeze and seg_frozen


def cache_key(cfg: Config) -> str:
    """Hash of every setting that affects per-frame features."""
    d, f, s = cfg.data, cfg.features, cfg.segmentation
    relevant = {
        "fps": d.target_fps, "resize": list(d.resize), "mean": list(d.normalize_mean),
        "std": list(d.normalize_std), "backbone": f.backbone, "pretrained": f.pretrained,
        "segmentation": None if not s.enabled else {
            "encoder": s.encoder, "encoder_weights": s.encoder_weights, "checkpoint": s.checkpoint,
            "input_size": list(s.input_size), "threshold": s.threshold, "soft_mask": s.soft_mask,
        },
    }
    return hashlib.sha1(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]


def window_frame_indices(start_sec: float, end_sec: float, fps: float, n: int) -> np.ndarray:
    """Global ``fps``-grid frame indices sampled for one window (see ``read_window``)."""
    first = int(np.ceil(start_sec * fps - 1e-6))
    last = int(np.ceil(end_sec * fps - 1e-6))  # exclusive
    grid = np.arange(first, max(last, first + 1))
    return grid[np.linspace(0, len(grid) - 1, num=n).round().astype(int)]


def iter_grid_frames(path: str | Path, fps: float, resize: Tuple[int, int]) -> Iterator[Tuple[int, np.ndarray]]:
    """Decode ``path`` once, yielding ``(k, rgb_frame)`` for every frame on the ``fps`` grid.

    Uses the same nearest-source-frame rule as :func:`wrestleai.data.video.extract_frames`.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0:
            raise IOError(f"Video reports invalid FPS: {path}")
        h, w = resize
        k, idx, half = 0, 0, 0.5 / src_fps
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / src_fps
            rgb = None
            while k / fps < t + half:
                if rgb is None:
                    small = cv2.resize(frame, (int(w), int(h)), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                yield k, rgb
                k += 1
            idx += 1
    finally:
        cap.release()


class FeatureCache:
    """Computes and stores per-frame features for one model/config combination."""

    def __init__(self, cfg: Config, model: torch.nn.Module, device: torch.device) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.dir = Path(cfg.paths.feature_cache_dir) / cache_key(cfg)

    def _file(self, video_path: str) -> Path:
        p = Path(video_path).resolve()
        digest = hashlib.sha1(str(p).encode()).hexdigest()[:8]
        return self.dir / f"{p.stem}-{digest}.npz"

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
                logger.info("Video changed since caching; recomputing features for %s", video_path)
                return {}
            return {int(k): v for k, v in zip(data["ks"], data["feats"])}

    def _save(self, video_path: str, feats: Dict[int, np.ndarray]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        ks = np.array(sorted(feats), dtype=np.int32)
        np.savez(self._file(video_path), ks=ks, feats=np.stack([feats[int(k)] for k in ks]),
                 stamp=self._stamp(video_path))

    @torch.no_grad()
    def _encode(self, frames: List[np.ndarray]) -> np.ndarray:
        x = frames_to_tensor(np.stack(frames), self.cfg.data.normalize_mean, self.cfg.data.normalize_std)
        x = self.model.preprocess(x.to(self.device))
        return self.model.features(x).float().cpu().numpy().astype(np.float16)

    def _compute(self, video_path: str, needed: Sequence[int]) -> Dict[int, np.ndarray]:
        """Run the CNN over the ``needed`` grid frames of one video (single sequential decode)."""
        self.model.eval()
        wanted = set(int(k) for k in needed)
        out: Dict[int, np.ndarray] = {}
        batch_k: List[int] = []
        batch_f: List[np.ndarray] = []
        last_frame: Tuple[int, np.ndarray] | None = None
        started, max_k = time.time(), max(wanted)
        for k, frame in iter_grid_frames(video_path, self.cfg.data.target_fps, tuple(self.cfg.data.resize)):
            last_frame = (k, frame)
            if k in wanted:
                batch_k.append(k)
                batch_f.append(frame)
                if len(batch_f) == FEATURE_BATCH:
                    out.update(zip(batch_k, self._encode(batch_f)))
                    batch_k, batch_f = [], []
                    if len(out) % (FEATURE_BATCH * 20) == 0:
                        logger.info("  %s: %d/%d frames encoded (%.0fs)", Path(video_path).name, len(out),
                                    len(wanted), time.time() - started)
            if k >= max_k:
                break
        if batch_f:
            out.update(zip(batch_k, self._encode(batch_f)))
        missing = wanted - set(out)
        if missing:
            # Windows can reach a few frames past the last decodable frame; reuse the last one.
            if last_frame is None:
                raise IOError(f"No frames decoded from {video_path}")
            fill = self._encode([last_frame[1]])[0]
            out.update({k: fill for k in missing})
        logger.info("Encoded %d frames of %s in %.0fs", len(wanted), Path(video_path).name, time.time() - started)
        return out

    def window_features(self, samples: Sequence[WindowSample]) -> torch.Tensor:
        """Return float32 features ``(N, frames_per_window, D)`` for ``samples`` (in order)."""
        fps, n = self.cfg.data.target_fps, self.cfg.data.frames_per_window
        indices = [window_frame_indices(s.start_sec, s.end_sec, fps, n) for s in samples]
        by_video: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            by_video[s.video_path].append(i)
        result: List[np.ndarray | None] = [None] * len(samples)
        for v_i, (video, sample_ids) in enumerate(sorted(by_video.items()), 1):
            needed = sorted({int(k) for i in sample_ids for k in indices[i]})
            feats = self._load(video)
            missing = [k for k in needed if k not in feats]
            if missing:
                logger.info("Video %d/%d %s: encoding %d frames (%d cached)", v_i, len(by_video),
                            Path(video).name, len(missing), len(needed) - len(missing))
                feats.update(self._compute(video, missing))
                self._save(video, feats)
            for i in sample_ids:
                result[i] = np.stack([feats[int(k)] for k in indices[i]])
        return torch.from_numpy(np.stack(result).astype(np.float32))  # type: ignore[arg-type]
