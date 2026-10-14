"""End-to-end smoke test: synthetic data -> label -> extract -> train -> evaluate -> infer via the CLI.

Runs fully offline on CPU (tiny backbone, no pretrained weights). It checks that every
stage runs and writes well-formed outputs; it does not assert model quality.
"""

import json
import time
from pathlib import Path

import pytest

from wrestleai.cli import main
from wrestleai.data.synthetic import make_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_train_evaluate_infer_end_to_end(tmp_path: Path) -> None:
    started = time.time()
    videos = tmp_path / "videos"
    pairs = make_synthetic_dataset(videos, num_videos=3, duration=16.0, fps=12.0, size=(64, 64), seed=1)
    common = [
        "--config", str(ROOT / "configs" / "test.yaml"),
        "--set", f"paths.db_path={tmp_path / 'e2e.db'}",
        "--set", f"paths.checkpoint_dir={tmp_path / 'ckpt'}",
        "--set", f"paths.output_dir={tmp_path / 'out'}",
        "--set", f"paths.processed_dir={tmp_path / 'processed'}",
        "--set", "training.epochs=2",
        "--log-level", "WARNING",
    ]

    assert main(["extract", str(videos), *common]) == 0
    # The labeler CLI adds a range non-interactively and rebuilds that video's windows.
    assert main(["label", str(pairs[0][0]), "--add", "12", "14", "highlight", *common]) == 0
    assert main(["train", "--run-name", "smoke", *common]) == 0

    run_dir = tmp_path / "ckpt" / "smoke"
    summary = json.loads((run_dir / "cv_summary.json").read_text())
    assert len(summary["folds"]) == 2 and (run_dir / "best.pt").exists()
    assert (tmp_path / "ckpt" / "LATEST").read_text() == "smoke"

    assert main(["evaluate", *common]) == 0
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    assert evaluation["mode"] == "out_of_fold"
    assert evaluation["n"] == sum(evaluation["support"].values()) == summary["num_windows"]
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix"]:
        assert key in evaluation
    assert (run_dir / "evaluation_confusion_matrix.csv").exists()

    assert main(["infer", str(pairs[1][0]), *common]) == 0
    result = json.loads((tmp_path / "out" / f"{pairs[1][0].stem}.highlights.json").read_text())
    assert len(result["windows"]) > 0 and isinstance(result["highlights"], list)
    assert all(0.0 <= w["p_highlight"] <= 1.0 for w in result["windows"])
    assert (tmp_path / "out" / f"{pairs[1][0].stem}.windows.csv").exists()

    assert time.time() - started < 60, "smoke test should finish in under a minute on CPU"
