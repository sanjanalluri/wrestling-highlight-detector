"""Inference: per-window highlight probabilities and merged highlight time ranges."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from wrestleai.data.dataset import WindowSample, make_dataloader
from wrestleai.data.video import probe_video
from wrestleai.data.windowing import Window, compute_windows
from wrestleai.training.checkpoint import load_checkpoint
from wrestleai.training.trainer import predict
from wrestleai.utils.runtime import resolve_device
from wrestleai.utils.timefmt import format_time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectedRange:
    """A merged span of positive windows."""

    start_sec: float
    end_sec: float
    max_prob: float
    mean_prob: float
    num_windows: int


def merge_positive_windows(
    windows: Sequence[Window],
    probs: Sequence[float],
    threshold: float = 0.5,
    max_gap: float = 0.0,
    min_duration: float = 0.0,
) -> List[DetectedRange]:
    """Merge windows with ``prob >= threshold`` into contiguous time ranges.

    Windows are merged when they overlap or touch, or are separated by at most
    ``max_gap`` seconds. Ranges shorter than ``min_duration`` are dropped.
    """
    if len(windows) != len(probs):
        raise ValueError("windows and probs must have the same length")
    order = sorted(range(len(windows)), key=lambda i: windows[i].start_sec)
    ranges: List[DetectedRange] = []
    cur_start: Optional[float] = None
    cur_end = 0.0
    cur_probs: List[float] = []

    def close() -> None:
        if cur_start is not None and cur_end - cur_start >= min_duration - 1e-9:
            ranges.append(DetectedRange(round(cur_start, 3), round(cur_end, 3), float(max(cur_probs)),
                                        float(np.mean(cur_probs)), len(cur_probs)))

    for i in order:
        if probs[i] < threshold:
            continue
        w = windows[i]
        if cur_start is not None and w.start_sec <= cur_end + max_gap + 1e-9:
            cur_end = max(cur_end, w.end_sec)
            cur_probs.append(float(probs[i]))
        else:
            close()
            cur_start, cur_end, cur_probs = w.start_sec, w.end_sec, [float(probs[i])]
    close()
    return ranges


def infer_video(
    video_path: str | Path,
    checkpoint: str | Path,
    out_dir: Optional[str | Path] = None,
    threshold: Optional[float] = None,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """Score every window of ``video_path`` and detect highlight ranges.

    Writes ``<stem>.windows.csv`` and ``<stem>.highlights.json`` to ``out_dir`` when given.
    """
    device = resolve_device(device_name)
    model, cfg, _ = load_checkpoint(checkpoint, device)
    info = probe_video(video_path)
    windows = compute_windows(info.duration_seconds, cfg.data.window_seconds, cfg.data.stride_seconds,
                              cfg.data.min_window_seconds)
    if not windows:
        raise RuntimeError(f"Video too short for a single window: {video_path}")
    samples = [WindowSample(str(video_path), w.start_sec, w.end_sec) for w in windows]
    probs = predict(model, make_dataloader(samples, cfg, shuffle=False, batch_size=cfg.inference.batch_size), device)
    positive = cfg.labels.classes.index(cfg.labels.positive_class)
    pos_probs = probs[:, positive]
    thr = cfg.inference.threshold if threshold is None else threshold
    ranges = merge_positive_windows(windows, pos_probs.tolist(), thr, cfg.inference.merge_gap_seconds,
                                    cfg.inference.min_range_seconds)
    result: Dict[str, Any] = {
        "video": str(video_path),
        "checkpoint": str(checkpoint),
        "threshold": thr,
        "classes": cfg.labels.classes,
        "positive_class": cfg.labels.positive_class,
        "windows": [
            {"index": w.index, "start_sec": w.start_sec, "end_sec": w.end_sec,
             f"p_{cfg.labels.positive_class}": float(pos_probs[i]),
             "predicted": cfg.labels.classes[int(probs[i].argmax())]}
            for i, w in enumerate(windows)
        ],
        "highlights": [asdict(r) for r in ranges],
    }
    logger.info("%s: %d windows, %d highlight range(s) at threshold %.2f", Path(video_path).name,
                len(windows), len(ranges), thr)
    for r in ranges:
        logger.info("  %s -> %s  (max p=%.3f over %d windows)", format_time(r.start_sec), format_time(r.end_sec),
                    r.max_prob, r.num_windows)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        with open(out / f"{stem}.windows.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(result["windows"][0].keys()))
            writer.writeheader()
            writer.writerows(result["windows"])
        with open(out / f"{stem}.highlights.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        logger.info("Wrote %s and %s", out / f"{stem}.windows.csv", out / f"{stem}.highlights.json")
    return result
