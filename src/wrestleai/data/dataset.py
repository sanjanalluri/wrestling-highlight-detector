"""PyTorch Dataset/DataLoader producing ``(frames, label)`` clip sequences."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wrestleai.config import Config
from wrestleai.data.video import read_window
from wrestleai.db.database import ClipWindowRecord, Database

logger = logging.getLogger(__name__)

UNLABELED = -1


@dataclass(frozen=True)
class WindowSample:
    """Everything needed to load one training/inference example."""

    video_path: str
    start_sec: float
    end_sec: float
    label: int = UNLABELED
    window_id: Optional[int] = None  # clip_windows.id when the sample comes from the DB
    video_id: Optional[int] = None


def frames_to_tensor(
    frames: np.ndarray, mean: Sequence[float], std: Sequence[float]
) -> torch.Tensor:
    """Convert ``uint8`` RGB ``(T, H, W, 3)`` to normalised float ``(T, 3, H, W)``."""
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).float().div_(255.0)
    tensor = tensor.permute(0, 3, 1, 2)
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean_t) / std_t


class ClipWindowDataset(Dataset):
    """Decodes ``frames_per_window`` frames for each window on demand.

    Items are ``(frames, label)`` with ``frames`` shaped ``(T, 3, H, W)``.
    """

    def __init__(self, samples: Sequence[WindowSample], cfg: Config) -> None:
        self.samples: List[WindowSample] = list(samples)
        self.fps = cfg.data.target_fps
        self.num_frames = cfg.data.frames_per_window
        self.resize: Tuple[int, int] = tuple(cfg.data.resize)  # type: ignore[assignment]
        self.mean = cfg.data.normalize_mean
        self.std = cfg.data.normalize_std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[idx]
        frames = read_window(s.video_path, s.start_sec, s.end_sec, self.fps, self.num_frames, self.resize)
        return frames_to_tensor(frames, self.mean, self.std), int(s.label)

    @property
    def labels(self) -> np.ndarray:
        return np.array([s.label for s in self.samples], dtype=np.int64)


def samples_from_db(
    db: Database,
    cfg: Config,
    video_ids: Optional[Sequence[int]] = None,
    window_ids: Optional[Sequence[int]] = None,
) -> List[WindowSample]:
    """Build labeled samples from the ``clip_windows`` table.

    Windows whose label is not in ``cfg.labels.classes`` are skipped with a warning.
    """
    class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(cfg.labels.classes)}
    records: List[ClipWindowRecord]
    if window_ids is not None:
        records = db.get_clip_windows_by_ids(window_ids)
    else:
        records = db.list_clip_windows(video_ids)
    paths: Dict[int, str] = {}
    samples: List[WindowSample] = []
    skipped = 0
    for rec in records:
        if rec.video_id not in paths:
            video = db.get_video(rec.video_id)
            if video is None:
                raise KeyError(f"clip window {rec.id} references missing video {rec.video_id}")
            paths[rec.video_id] = video.read_path
        if rec.label not in class_to_idx:
            skipped += 1
            continue
        samples.append(
            WindowSample(paths[rec.video_id], rec.start_sec, rec.end_sec,
                         class_to_idx[rec.label], rec.id, rec.video_id)
        )
    if skipped:
        logger.warning("Skipped %d windows with labels outside %s", skipped, cfg.labels.classes)
    return samples


def make_dataloader(
    samples: Sequence[WindowSample], cfg: Config, shuffle: bool, batch_size: Optional[int] = None
) -> DataLoader:
    """Wrap samples in a :class:`ClipWindowDataset` and a :class:`DataLoader`."""
    return DataLoader(
        ClipWindowDataset(samples, cfg),
        batch_size=batch_size or cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=cfg.training.num_workers,
        drop_last=False,
    )
