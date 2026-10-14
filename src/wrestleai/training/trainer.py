"""K-fold cross-validated training with class weighting, checkpointing and early stopping."""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wrestleai.config import Config, save_config
from wrestleai.data.dataset import WindowSample, make_dataloader, samples_from_db
from wrestleai.db.database import Database
from wrestleai.models.highlight_model import HighlightModel, build_model
from wrestleai.training.checkpoint import LATEST_FILE, save_checkpoint
from wrestleai.training.metrics import compute_metrics, format_metrics
from wrestleai.training.splits import make_folds
from wrestleai.utils.runtime import resolve_device, set_seed

logger = logging.getLogger(__name__)


def class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights ``N / (C * n_c)``; classes absent from ``labels`` get 1.0."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    weights = np.ones(num_classes)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict(model: HighlightModel, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Return softmax probabilities ``(N, C)`` for every batch in ``loader`` (in order)."""
    model.eval()
    out: List[np.ndarray] = []
    for clips, _ in loader:
        out.append(torch.softmax(model(clips.to(device)), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.head.num_classes))


class EarlyStopping:
    """Tracks the best value of a maximised metric and signals when to stop."""

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Record ``value``; return ``True`` if it is a new best."""
        if value > self.best + self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


@dataclass
class FoldResult:
    """Outcome of training one fold."""

    fold: int
    best_epoch: int
    val_metrics: Dict[str, Any]
    checkpoint: str
    best_score: float
    history: List[Dict[str, float]] = field(default_factory=list)


def _monitor_value(metrics: Dict[str, Any], name: str, val_loss: float) -> float:
    if name == "loss":
        return -val_loss
    value = metrics.get(name)
    if value is None:  # e.g. AUC undefined on a single-class validation fold
        return -val_loss
    return float(value)


def train_fold(
    cfg: Config,
    train_samples: Sequence[WindowSample],
    val_samples: Sequence[WindowSample],
    fold: int,
    run_dir: Path,
    device: torch.device,
) -> FoldResult:
    """Train one model on ``train_samples`` and keep the epoch best on ``val_samples``."""
    tc = cfg.training
    set_seed(tc.seed + fold)
    model = build_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tc.learning_rate, weight_decay=tc.weight_decay)
    train_labels = [s.label for s in train_samples]
    weight = class_weights(train_labels, cfg.labels.num_classes).to(device) if tc.class_weighting else None
    if weight is not None:
        logger.info("Fold %d class weights: %s", fold, [round(float(w), 3) for w in weight])
    criterion = nn.CrossEntropyLoss(weight=weight)
    train_loader = make_dataloader(train_samples, cfg, shuffle=True)
    val_loader = make_dataloader(val_samples, cfg, shuffle=False)
    val_y = np.array([s.label for s in val_samples])
    positive = cfg.labels.classes.index(cfg.labels.positive_class)
    stopper = EarlyStopping(tc.early_stopping_patience, tc.early_stopping_min_delta)
    ckpt_path = run_dir / f"fold_{fold}.pt"
    history: List[Dict[str, float]] = []
    best_epoch, best_metrics = 0, {}
    for epoch in range(1, tc.epochs + 1):
        start = time.time()
        model.train()
        running, seen = 0.0, 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(clips), labels)
            loss.backward()
            optimizer.step()
            running += float(loss) * len(labels)
            seen += len(labels)
        train_loss = running / max(seen, 1)
        probs = predict(model, val_loader, device)
        val_loss = float(criterion(torch.log(torch.from_numpy(probs).float().clamp_min(1e-8)).to(device),
                                   torch.from_numpy(val_y).to(device)))
        metrics = compute_metrics(val_y, probs, cfg.labels.classes, positive, cfg.inference.threshold)
        score = _monitor_value(metrics, tc.monitor, val_loss)
        improved = stopper.step(score)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_f1": metrics["f1"], "val_accuracy": metrics["accuracy"]})
        logger.info(
            "Fold %d epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  val_f1=%.3f%s  (%.1fs)",
            fold, epoch, tc.epochs, train_loss, val_loss, metrics["accuracy"], metrics["f1"],
            "  *best*" if improved else "", time.time() - start,
        )
        if improved:
            best_epoch, best_metrics = epoch, metrics
            save_checkpoint(
                ckpt_path, model, cfg, fold=fold, epoch=epoch, val_metrics=metrics,
                train_window_ids=[s.window_id for s in train_samples],
                val_window_ids=[s.window_id for s in val_samples],
            )
        if stopper.should_stop:
            logger.info("Fold %d early stopping at epoch %d (best epoch %d)", fold, epoch, best_epoch)
            break
    logger.info("Fold %d best epoch %d validation metrics:\n%s", fold, best_epoch, format_metrics(best_metrics))
    return FoldResult(fold, best_epoch, best_metrics, str(ckpt_path), stopper.best, history)


def run_training(
    cfg: Config,
    db: Database,
    run_name: Optional[str] = None,
    video_ids: Optional[Sequence[int]] = None,
) -> Path:
    """Run k-fold cross-validation over all labeled windows; return the run directory.

    The run directory holds ``fold_<k>.pt`` checkpoints (each recording its train/val
    window ids), ``best.pt`` (the fold with the best monitored validation score),
    ``config.yaml`` and ``cv_summary.json``.
    """
    tc = cfg.training
    set_seed(tc.seed)
    device = resolve_device(tc.device)
    samples = samples_from_db(db, cfg, video_ids=video_ids)
    if not samples:
        raise RuntimeError("No labeled clip windows found; run `wrestleai extract` first")
    labels = np.array([s.label for s in samples])
    groups = np.array([s.video_id for s in samples])
    logger.info(
        "Training on %d windows from %d videos; class counts %s; device=%s",
        len(samples), len(np.unique(groups)),
        dict(zip(cfg.labels.classes, np.bincount(labels, minlength=cfg.labels.num_classes).tolist())), device,
    )
    folds = make_folds(labels, groups, tc.k_folds, tc.seed, tc.group_by_video)
    run_name = run_name or time.strftime("run-%Y%m%d-%H%M%S")
    run_dir = Path(cfg.paths.checkpoint_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    results: List[FoldResult] = []
    for k, (train_idx, val_idx) in enumerate(folds):
        logger.info("=== Fold %d/%d: %d train / %d val windows ===", k + 1, len(folds), len(train_idx), len(val_idx))
        results.append(train_fold(cfg, [samples[i] for i in train_idx], [samples[i] for i in val_idx],
                                  k, run_dir, device))
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    summary: Dict[str, Any] = {"run": run_name, "num_windows": len(samples), "folds": [], "mean": {}, "std": {}}
    for r in results:
        summary["folds"].append({"fold": r.fold, "best_epoch": r.best_epoch, "checkpoint": r.checkpoint,
                                 "val_metrics": r.val_metrics, "history": r.history})
    for key in keys:
        values = [r.val_metrics.get(key) for r in results if r.val_metrics.get(key) is not None]
        summary["mean"][key] = float(np.mean(values)) if values else None
        summary["std"][key] = float(np.std(values)) if values else None
    best = max(results, key=lambda r: r.best_score)
    shutil.copyfile(best.checkpoint, run_dir / "best.pt")
    summary["best_fold"] = best.fold
    with open(run_dir / "cv_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    (Path(cfg.paths.checkpoint_dir) / LATEST_FILE).write_text(run_name, encoding="utf-8")
    logger.info(
        "Cross-validation finished (%d folds). Mean val metrics: %s. best.pt = fold %d. Run dir: %s",
        len(results), {k: (None if v is None else round(v, 4)) for k, v in summary["mean"].items()},
        best.fold, run_dir,
    )
    return run_dir
