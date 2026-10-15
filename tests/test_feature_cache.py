"""Feature caching for frozen backbones (random weights, no downloads)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wrestleai.cli import main
from wrestleai.data.dataset import WindowSample, frames_to_tensor
from wrestleai.data.video import read_window
from wrestleai.data.windowing import compute_windows
from wrestleai.features.cache import FeatureCache, cache_key, uses_frozen_features, window_frame_indices
from wrestleai.models import build_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def frozen_cfg(cfg):
    cfg.features.freeze = True
    return cfg


def _samples(video: Path, cfg) -> list:
    windows = compute_windows(16.0, cfg.data.window_seconds, cfg.data.stride_seconds, cfg.data.min_window_seconds)
    return [WindowSample(str(video), w.start_sec, w.end_sec) for w in windows]


def test_uses_frozen_features(cfg) -> None:
    assert not uses_frozen_features(cfg)  # test config trains the backbone
    cfg.features.freeze = True
    assert uses_frozen_features(cfg)
    cfg.training.cache_features = False
    assert not uses_frozen_features(cfg)
    cfg.training.cache_features, cfg.segmentation.enabled, cfg.segmentation.freeze = True, True, False
    assert not uses_frozen_features(cfg)


def test_window_frame_indices() -> None:
    assert window_frame_indices(0.0, 2.0, 6.0, 4).tolist() == [0, 4, 7, 11]
    assert window_frame_indices(1.0, 3.0, 6.0, 4).tolist() == [6, 10, 13, 17]


def test_cached_features_match_dataset_path(frozen_cfg, synthetic_videos) -> None:
    torch.manual_seed(0)
    model = build_model(frozen_cfg).eval()
    samples = _samples(synthetic_videos[0][0], frozen_cfg)
    cached = FeatureCache(frozen_cfg, model, torch.device("cpu")).window_features(samples)
    d = frozen_cfg.data
    assert cached.shape == (len(samples), d.frames_per_window, model.features.out_dim)
    for i in [0, len(samples) // 2, len(samples) - 1]:
        frames = read_window(samples[i].video_path, samples[i].start_sec, samples[i].end_sec,
                             d.target_fps, d.frames_per_window, tuple(d.resize))
        with torch.no_grad():
            direct = model.features(frames_to_tensor(frames, d.normalize_mean, d.normalize_std))
        assert torch.allclose(cached[i], direct, atol=2e-2, rtol=1e-2)


def test_second_call_reads_from_disk(frozen_cfg, synthetic_videos, monkeypatch) -> None:
    model = build_model(frozen_cfg).eval()
    samples = _samples(synthetic_videos[1][0], frozen_cfg)
    first = FeatureCache(frozen_cfg, model, torch.device("cpu")).window_features(samples)

    def boom(*args, **kwargs):
        raise AssertionError("features should come from the cache")

    monkeypatch.setattr(FeatureCache, "_compute", boom)
    second = FeatureCache(frozen_cfg, model, torch.device("cpu")).window_features(samples)
    assert torch.equal(first, second)


def test_cache_key_changes_with_feature_settings(frozen_cfg) -> None:
    key = cache_key(frozen_cfg)
    frozen_cfg.training.epochs += 5  # training settings don't affect features
    assert cache_key(frozen_cfg) == key
    frozen_cfg.data.resize = (32, 32)
    assert cache_key(frozen_cfg) != key


def test_frozen_backbone_end_to_end(tmp_path: Path) -> None:
    from wrestleai.data.synthetic import make_synthetic_dataset

    videos = tmp_path / "videos"
    pairs = make_synthetic_dataset(videos, num_videos=3, duration=12.0, fps=12.0, size=(64, 64), seed=2)
    common = [
        "--config", str(ROOT / "configs" / "test.yaml"),
        "--set", f"paths.db_path={tmp_path / 'e2e.db'}",
        "--set", f"paths.checkpoint_dir={tmp_path / 'ckpt'}",
        "--set", f"paths.output_dir={tmp_path / 'out'}",
        "--set", f"paths.feature_cache_dir={tmp_path / 'cache'}",
        "--set", "features.freeze=true",
        "--set", "training.epochs=3",
        "--log-level", "WARNING",
    ]
    assert main(["extract", str(videos), *common]) == 0
    assert main(["train", "--run-name", "frozen", *common]) == 0
    assert list((tmp_path / "cache").rglob("*.npz")), "features should have been cached"
    assert main(["evaluate", *common]) == 0
    evaluation = json.loads((tmp_path / "ckpt" / "frozen" / "evaluation.json").read_text())
    assert evaluation["n"] == json.loads((tmp_path / "ckpt" / "frozen" / "cv_summary.json").read_text())["num_windows"]
    assert main(["infer", str(pairs[0][0]), *common]) == 0
    result = json.loads((tmp_path / "out" / f"{pairs[0][0].stem}.highlights.json").read_text())
    assert all(0.0 <= w["p_highlight"] <= 1.0 for w in result["windows"])
