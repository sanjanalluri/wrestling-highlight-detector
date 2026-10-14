"""Shared pytest fixtures. Everything here runs offline on CPU."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest

from wrestleai.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_CONFIG = REPO_ROOT / "configs" / "test.yaml"


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """The tiny offline test config with all paths redirected into ``tmp_path``."""
    return load_config(
        TEST_CONFIG,
        [
            f"paths.db_path={tmp_path / 'test.db'}",
            f"paths.video_dir={tmp_path / 'videos'}",
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.checkpoint_dir={tmp_path / 'checkpoints'}",
            f"paths.output_dir={tmp_path / 'outputs'}",
        ],
    )


@pytest.fixture(scope="session")
def synthetic_videos(tmp_path_factory: pytest.TempPathFactory) -> List[Tuple[Path, Path]]:
    """Three short synthetic videos with label CSVs, generated once per session."""
    from wrestleai.data.synthetic import make_synthetic_dataset

    out = tmp_path_factory.mktemp("synthetic")
    return make_synthetic_dataset(out, num_videos=3, duration=16.0, fps=12.0, size=(64, 64), seed=0)
