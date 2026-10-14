"""Saving/loading model checkpoints and locating training runs."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from wrestleai.config import Config, config_from_dict
from wrestleai.models.highlight_model import HighlightModel, build_model

logger = logging.getLogger(__name__)

LATEST_FILE = "LATEST"


def save_checkpoint(path: str | Path, model: HighlightModel, cfg: Config, **extra: Any) -> None:
    """Save the full model state plus the config needed to rebuild it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg.to_dict(), **extra}, path)


def load_checkpoint(
    path: str | Path, device: Optional[torch.device] = None
) -> Tuple[HighlightModel, Config, Dict[str, Any]]:
    """Rebuild the model from a checkpoint without downloading any pretrained weights.

    Returns ``(model_in_eval_mode, config, checkpoint_dict)``.
    """
    ckpt = torch.load(Path(path), map_location="cpu")
    cfg = config_from_dict(ckpt["config"])
    build_cfg = copy.deepcopy(cfg)
    # All weights come from the checkpoint, so skip pretrained downloads when rebuilding.
    build_cfg.features.pretrained = False
    build_cfg.segmentation.encoder_weights = None
    build_cfg.segmentation.checkpoint = None
    previous = logging.getLogger("wrestleai.segmentation.deeplab").level
    logging.getLogger("wrestleai.segmentation.deeplab").setLevel(logging.ERROR)
    try:
        model = build_model(build_cfg)
    finally:
        logging.getLogger("wrestleai.segmentation.deeplab").setLevel(previous)
    model.load_state_dict(ckpt["model_state"])
    model.to(device or torch.device("cpu")).eval()
    return model, cfg, ckpt


def resolve_run_dir(checkpoint_dir: str | Path, run: Optional[str | Path] = None) -> Path:
    """Return the run directory: ``run`` if given (name or path), else the latest run."""
    base = Path(checkpoint_dir)
    if run is not None:
        candidate = Path(run)
        if candidate.is_dir():
            return candidate
        if (base / run).is_dir():
            return base / run
        raise FileNotFoundError(f"Run not found: {run} (looked in {base})")
    latest = base / LATEST_FILE
    if not latest.exists():
        raise FileNotFoundError(f"No training runs in {base}; run `wrestleai train` first")
    return base / latest.read_text(encoding="utf-8").strip()
