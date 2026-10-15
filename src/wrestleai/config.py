"""Typed, YAML-backed configuration for the Groot pipeline.

Every hyperparameter lives in a dataclass below. ``configs/default.yaml`` mirrors
these defaults; any YAML file only needs to specify the keys it overrides, and
individual values can be overridden from the CLI with ``--set section.key=value``.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PathsConfig:
    """Filesystem locations. Relative paths are resolved against the working directory."""

    db_path: str = "data/wrestleai.db"
    video_dir: str = "match_directory"
    processed_dir: str = "data/processed_videos"
    checkpoint_dir: str = "checkpoints"
    feature_cache_dir: str = "data/feature_cache"
    output_dir: str = "outputs"


@dataclass
class DataConfig:
    """Frame extraction and clip windowing."""

    target_fps: float = 12.0
    resize: Tuple[int, int] = (224, 224)  # (height, width) fed to the CNN
    window_seconds: float = 5.0
    stride_seconds: float = 2.5
    frames_per_window: int = 16  # uniformly sampled from the window at target_fps
    min_window_seconds: float = 1.0  # trailing partial windows shorter than this are dropped
    supported_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
    )
    # ImageNet statistics used by both the pretrained segmentation encoder and the CNN.
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class LabelsConfig:
    """Label vocabulary. Index 0 is the default (background) class for unlabeled time."""

    classes: List[str] = field(default_factory=lambda: ["non_highlight", "highlight"])
    positive_class: str = "highlight"  # used for binary metrics and inference ranges
    # A window takes a range's label when that range covers at least this fraction of it.
    overlap_threshold: float = 0.5

    @property
    def default_class(self) -> str:
        return self.classes[0]

    @property
    def num_classes(self) -> int:
        return len(self.classes)


@dataclass
class SegmentationConfig:
    """DeepLabV3+ silhouette masking applied before feature extraction."""

    enabled: bool = False
    encoder: str = "resnet50"
    encoder_weights: Optional[str] = "imagenet"
    # Fine-tuned person-segmentation weights. Without this, only the encoder is pretrained
    # and the decoder is randomly initialised, so masks are not meaningful.
    checkpoint: Optional[str] = None
    input_size: Tuple[int, int] = (512, 512)
    threshold: float = 0.5
    soft_mask: bool = False  # multiply by probabilities instead of a hard 0/1 mask
    freeze: bool = True


@dataclass
class FeaturesConfig:
    """Per-frame CNN backbone (any timm model name)."""

    backbone: str = "efficientnet_b3"
    pretrained: bool = True
    freeze: bool = True


@dataclass
class TemporalConfig:
    """Bidirectional LSTM over the per-frame feature sequence."""

    hidden_size: int = 384
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    pooling: str = "mean"  # "mean" over time or "last" hidden state


@dataclass
class ClassifierConfig:
    """Dense classification head."""

    hidden_dims: List[int] = field(default_factory=lambda: [512, 128])
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    """Cross-validated training loop."""

    k_folds: int = 5
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    class_weighting: bool = True
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-3
    monitor: str = "f1"  # validation metric used for checkpoint selection / early stopping
    group_by_video: bool = True  # keep all windows of a video in the same fold
    seed: int = 42
    num_workers: int = 0
    # With a frozen backbone (and frozen/disabled segmentation), compute per-frame CNN
    # features once, cache them under paths.feature_cache_dir, and train only the Bi-LSTM
    # and classifier on them. Much faster on CPU; results are identical in expectation.
    cache_features: bool = True
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"


@dataclass
class InferenceConfig:
    """Window scoring and range merging."""

    threshold: float = 0.5
    merge_gap_seconds: float = 0.0  # merge positive ranges separated by at most this gap
    min_range_seconds: float = 0.0
    batch_size: int = 8


@dataclass
class Config:
    """Root configuration object."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain, YAML/JSON-serialisable dict."""
        return _to_plain(dataclasses.asdict(self))

    def validate(self) -> None:
        """Raise ``ValueError`` on inconsistent settings."""
        d, lab, t = self.data, self.labels, self.training
        if d.target_fps <= 0:
            raise ValueError("data.target_fps must be positive")
        if d.window_seconds <= 0 or d.stride_seconds <= 0:
            raise ValueError("data.window_seconds and data.stride_seconds must be positive")
        if d.frames_per_window < 1:
            raise ValueError("data.frames_per_window must be >= 1")
        if len(lab.classes) < 2 or len(set(lab.classes)) != len(lab.classes):
            raise ValueError("labels.classes needs at least two unique class names")
        if lab.positive_class not in lab.classes:
            raise ValueError(f"labels.positive_class {lab.positive_class!r} not in labels.classes")
        if lab.positive_class == lab.default_class:
            raise ValueError("labels.positive_class must differ from the default class (classes[0])")
        if not 0.0 < lab.overlap_threshold <= 1.0:
            raise ValueError("labels.overlap_threshold must be in (0, 1]")
        if t.k_folds < 2:
            raise ValueError("training.k_folds must be >= 2")
        if self.temporal.pooling not in {"mean", "last"}:
            raise ValueError("temporal.pooling must be 'mean' or 'last'")


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _build(cls: type, values: Dict[str, Any], prefix: str) -> Any:
    """Instantiate dataclass ``cls`` from ``values``, recursing into nested dataclasses."""
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown config key(s) under '{prefix or 'root'}': {sorted(unknown)}")
    kwargs: Dict[str, Any] = {}
    defaults = cls()
    for name, f in known.items():
        if name not in values:
            continue
        raw = values[name]
        current = getattr(defaults, name)
        if dataclasses.is_dataclass(current):
            if not isinstance(raw, dict):
                raise ValueError(f"Config section '{prefix}{name}' must be a mapping")
            kwargs[name] = _build(type(current), raw, f"{prefix}{name}.")
        elif isinstance(current, tuple) and isinstance(raw, list):
            kwargs[name] = tuple(raw)
        else:
            kwargs[name] = raw
    return cls(**kwargs)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_overrides(overrides: Sequence[str]) -> Dict[str, Any]:
    """Turn ``["training.epochs=3", "features.pretrained=false"]`` into a nested dict.

    Values are parsed as YAML scalars, so numbers, booleans, ``null`` and lists work.
    """
    result: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like section.key=value")
        dotted, raw_value = item.split("=", 1)
        node = result
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw_value)
    return result


def load_config(
    path: Optional[str | Path] = None, overrides: Optional[Sequence[str]] = None
) -> Config:
    """Load a config from YAML (optional) and apply ``--set`` style overrides."""
    values: Dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            values = yaml.safe_load(fh) or {}
        if not isinstance(values, dict):
            raise ValueError(f"Config file {path} must contain a mapping at the top level")
        logger.debug("Loaded config from %s", path)
    if overrides:
        values = _deep_merge(values, parse_overrides(overrides))
    cfg = _build(Config, values, "")
    cfg.validate()
    return cfg


def config_from_dict(values: Dict[str, Any]) -> Config:
    """Rebuild a :class:`Config` from :meth:`Config.to_dict` output (e.g. a checkpoint)."""
    cfg = _build(Config, values, "")
    cfg.validate()
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Write ``cfg`` to YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
