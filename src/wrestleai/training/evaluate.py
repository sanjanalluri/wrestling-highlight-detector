"""Evaluation: out-of-fold metrics for a CV run, or one checkpoint on chosen videos."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from wrestleai.config import Config
from wrestleai.data.dataset import samples_from_db
from wrestleai.db.database import Database
from wrestleai.training.checkpoint import load_checkpoint
from wrestleai.training.metrics import compute_metrics, format_metrics
from wrestleai.training.trainer import score_samples
from wrestleai.utils.runtime import resolve_device

logger = logging.getLogger(__name__)


def _write_outputs(metrics: Dict[str, Any], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with open(out_dir / f"{stem}_confusion_matrix.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred", *metrics["classes"]])
        for cls, row in zip(metrics["classes"], metrics["confusion_matrix"]):
            writer.writerow([cls, *row])


def evaluate_run(run_dir: str | Path, db: Database, device_name: str = "auto") -> Dict[str, Any]:
    """Out-of-fold evaluation of a cross-validation run.

    Each ``fold_<k>.pt`` scores exactly the windows it was validated on, so every
    labeled window is predicted by a model that never trained on it. Results are written
    to ``<run_dir>/evaluation.json`` and ``evaluation_confusion_matrix.csv``.
    """
    run_dir = Path(run_dir)
    fold_paths = sorted(run_dir.glob("fold_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not fold_paths:
        raise FileNotFoundError(f"No fold checkpoints in {run_dir}")
    device = resolve_device(device_name)
    all_y: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    per_fold: List[Dict[str, Any]] = []
    cfg: Optional[Config] = None
    for path in fold_paths:
        model, cfg, ckpt = load_checkpoint(path, device)
        samples = samples_from_db(db, cfg, window_ids=ckpt["val_window_ids"])
        probs = score_samples(model, samples, cfg, device)
        y = np.array([s.label for s in samples])
        positive = cfg.labels.classes.index(cfg.labels.positive_class)
        fold_metrics = compute_metrics(y, probs, cfg.labels.classes, positive, cfg.inference.threshold)
        per_fold.append({"fold": ckpt.get("fold"), "epoch": ckpt.get("epoch"), **fold_metrics})
        logger.info("Fold %s (%d windows):\n%s", ckpt.get("fold"), len(y), format_metrics(fold_metrics))
        all_y.append(y)
        all_p.append(probs)
    assert cfg is not None
    positive = cfg.labels.classes.index(cfg.labels.positive_class)
    overall = compute_metrics(np.concatenate(all_y), np.concatenate(all_p), cfg.labels.classes,
                              positive, cfg.inference.threshold)
    overall["mode"] = "out_of_fold"
    overall["per_fold"] = per_fold
    _write_outputs(overall, run_dir, "evaluation")
    logger.info("Out-of-fold metrics over %d windows:\n%s", overall["n"], format_metrics(overall))
    return overall


def evaluate_checkpoint(
    checkpoint: str | Path,
    db: Database,
    video_ids: Sequence[int],
    out_dir: str | Path,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """Score one checkpoint on all windows of ``video_ids`` (e.g. held-out matches).

    Only meaningful when those videos were not used for training that checkpoint.
    """
    device = resolve_device(device_name)
    model, cfg, ckpt = load_checkpoint(checkpoint, device)
    samples = samples_from_db(db, cfg, video_ids=video_ids)
    if not samples:
        raise RuntimeError(f"No labeled windows for video ids {list(video_ids)}")
    trained_on = set(ckpt.get("train_window_ids") or [])
    overlap = sum(1 for s in samples if s.window_id in trained_on)
    if overlap:
        logger.warning("%d of %d evaluation windows were in this checkpoint's training set", overlap, len(samples))
    probs = score_samples(model, samples, cfg, device)
    y = np.array([s.label for s in samples])
    positive = cfg.labels.classes.index(cfg.labels.positive_class)
    metrics = compute_metrics(y, probs, cfg.labels.classes, positive, cfg.inference.threshold)
    metrics.update({"mode": "checkpoint", "checkpoint": str(checkpoint), "video_ids": list(video_ids),
                    "windows_seen_in_training": overlap})
    _write_outputs(metrics, Path(out_dir), f"evaluation_{Path(checkpoint).stem}")
    logger.info("Checkpoint %s on %d windows:\n%s", checkpoint, len(y), format_metrics(metrics))
    return metrics
