"""Optical-flow motion descriptors."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from wrestleai.cli import main
from wrestleai.config import load_config
from wrestleai.data.dataset import WindowSample
from wrestleai.data.labels import load_label_file
from wrestleai.data.windowing import compute_windows
from wrestleai.features.cache import FeatureCache
from wrestleai.features.motion import MOTION_DIM, MotionCache, motion_descriptor
from wrestleai.models import build_model

ROOT = Path(__file__).resolve().parents[1]


def _texture(seed: int = 0, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 255, (size, size)).astype(np.uint8), (5, 5), 0)


def test_static_frames_have_no_motion() -> None:
    frame = _texture()
    d = motion_descriptor(frame, frame)
    assert d.shape == (MOTION_DIM,) and d.dtype == np.float32
    assert np.all(d[:6] < 0.15)  # magnitude stats: only sub-pixel solver noise
    assert np.all(d[6:8] == 0)  # no pixel moves more than 1 px


def test_camera_pan_is_separated_from_action() -> None:
    base = _texture(1, 128)
    pan_a, pan_b = base[16:112, 16:112], base[16:112, 20:116]  # whole view shifts 4 px
    pan = motion_descriptor(pan_a, pan_b)
    action_b = pan_a.copy()
    action_b[30:60, 30:60] = np.roll(pan_a[30:60, 30:60], 6, axis=1)  # one region moves
    action = motion_descriptor(pan_a, action_b)
    assert pan[8] > 1.0  # camera magnitude (log1p of ~4 px)
    assert pan[0] < action[0]  # residual motion: action > pure camera pan
    assert action[8] < pan[8]


def test_motion_requires_cached_frozen_features() -> None:
    with pytest.raises(ValueError, match="motion"):
        load_config(ROOT / "configs" / "test.yaml", ["features.motion=true"])  # test config: backbone trained
    cfg = load_config(ROOT / "configs" / "test.yaml", ["features.motion=true", "features.freeze=true"])
    assert cfg.features.motion


def test_highlight_windows_move_more(cfg, synthetic_videos) -> None:
    cfg.features.freeze, cfg.features.motion = True, True
    video, labels_csv = synthetic_videos[0]
    ranges = load_label_file(labels_csv)
    windows = compute_windows(16.0, 2.0, 1.0, 1.0)
    samples = [WindowSample(str(video), w.start_sec, w.end_sec) for w in windows]
    motion = MotionCache(cfg).window_features(samples)
    assert motion.shape == (len(samples), cfg.data.frames_per_window, MOTION_DIM)
    inside = [i for i, w in enumerate(windows) if any(r.start_sec <= w.start_sec and w.end_sec <= r.end_sec for r in ranges)]
    outside = [i for i, w in enumerate(windows)
               if all(w.end_sec <= r.start_sec or w.start_sec >= r.end_sec for r in ranges)]
    assert inside and outside
    assert motion[inside, :, 0].mean() > motion[outside, :, 0].mean()

    model = build_model(cfg).eval()
    combined = FeatureCache(cfg, model, torch.device("cpu")).window_features(samples)
    assert combined.shape[-1] == model.features.out_dim + MOTION_DIM
    assert torch.equal(combined[..., -MOTION_DIM:], motion)
    with pytest.raises(RuntimeError, match="consecutive frames"):
        model(torch.zeros(1, cfg.data.frames_per_window, 3, *cfg.data.resize))


def test_motion_end_to_end(tmp_path: Path) -> None:
    from wrestleai.data.synthetic import make_synthetic_dataset

    videos = tmp_path / "videos"
    pairs = make_synthetic_dataset(videos, num_videos=3, duration=12.0, fps=12.0, size=(64, 64), seed=3)
    common = [
        "--config", str(ROOT / "configs" / "test.yaml"),
        "--set", f"paths.db_path={tmp_path / 'm.db'}",
        "--set", f"paths.checkpoint_dir={tmp_path / 'ckpt'}",
        "--set", f"paths.output_dir={tmp_path / 'out'}",
        "--set", f"paths.feature_cache_dir={tmp_path / 'cache'}",
        "--set", "features.freeze=true", "--set", "features.motion=true",
        "--set", "training.epochs=2", "--log-level", "WARNING",
    ]
    assert main(["extract", str(videos), *common]) == 0
    assert main(["train", "--run-name", "m", *common]) == 0
    assert main(["evaluate", *common]) == 0
    assert json.loads((tmp_path / "ckpt" / "m" / "evaluation.json").read_text())["n"] > 0
    assert main(["infer", str(pairs[0][0]), *common]) == 0
    assert list((tmp_path / "cache").glob("motion-*/*.npz"))
