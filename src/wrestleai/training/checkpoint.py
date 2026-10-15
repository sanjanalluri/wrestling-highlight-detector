"""Saving/loading model checkpoints and locating training runs."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from wrestleai.config import Config, config_from_dict
from wrestleai.models.highlight_model import HighlightModel, build_model

logger = logging.getLogger(__name__)

LATEST_FILE = "LATEST"


def frozen_prefixes(cfg: Config) -> List[str]:
    """State-dict prefixes that can be rebuilt from their source instead of being saved.

    A frozen, pretrained backbone is identical to the published weights, and a frozen
    segmenter loaded from ``segmentation.checkpoint`` is identical to that file, so
    checkpoints only need the trained layers.
    """
    prefixes = []
    if cfg.features.freeze and cfg.features.pretrained:
        prefixes.append("features.")
    if cfg.segmentation.enabled and cfg.segmentation.freeze and cfg.segmentation.checkpoint:
        prefixes.append("segmenter.")
    return prefixes


def save_checkpoint(path: str | Path, model: HighlightModel, cfg: Config, **extra: Any) -> None:
    """Save the trained weights plus the config needed to rebuild the model.

    Frozen pretrained parts (see :func:`frozen_prefixes`) are left out and reloaded from
    their source when the checkpoint is loaded.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    excluded = frozen_prefixes(cfg)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith(tuple(excluded))}
    torch.save({"model_state": state, "excluded_prefixes": excluded, "config": cfg.to_dict(), **extra}, path)


def load_checkpoint(
    path: str | Path, device: Optional[torch.device] = None
) -> Tuple[HighlightModel, Config, Dict[str, Any]]:
    """Rebuild the model from a checkpoint.

    Parts saved in the checkpoint are not downloaded again. Frozen pretrained parts that
    were left out are rebuilt from their source (the timm weights, cached locally after
    the first download, or the segmentation checkpoint file).

    Returns ``(model_in_eval_mode, config, checkpoint_dict)``.
    """
    ckpt = torch.load(Path(path), map_location="cpu")
    cfg = config_from_dict(ckpt["config"])
    excluded: List[str] = ckpt.get("excluded_prefixes", [])
    build_cfg = copy.deepcopy(cfg)
    if "features." not in excluded:
        build_cfg.features.pretrained = False
    if "segmenter." not in excluded:
        build_cfg.segmentation.encoder_weights = None
        build_cfg.segmentation.checkpoint = None
    previous = logging.getLogger("wrestleai.segmentation.deeplab").level
    logging.getLogger("wrestleai.segmentation.deeplab").setLevel(logging.ERROR)
    try:
        model = build_model(build_cfg)
    finally:
        logging.getLogger("wrestleai.segmentation.deeplab").setLevel(previous)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    bad_missing = [k for k in missing if not k.startswith(tuple(excluded))]
    if bad_missing or unexpected:
        raise RuntimeError(f"Checkpoint {path} does not match the model: missing={bad_missing[:5]} "
                           f"unexpected={list(unexpected)[:5]}")
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
