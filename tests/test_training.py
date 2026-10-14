import numpy as np
import pytest
import torch

from wrestleai.training.metrics import compute_metrics
from wrestleai.training.splits import make_folds
from wrestleai.training.trainer import EarlyStopping, class_weights


def test_binary_metrics_match_hand_computation() -> None:
    y = [0, 0, 1, 1, 1]
    p1 = np.array([0.1, 0.6, 0.7, 0.4, 0.9])
    m = compute_metrics(y, np.stack([1 - p1, p1], 1), ["no", "yes"])
    # preds: 0,1,1,0,1 -> TP=2 FP=1 FN=1 TN=1
    assert m["confusion_matrix"] == [[1, 1], [1, 2]]
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(5 / 6)
    assert 0 < m["pr_auc"] <= 1


def test_auc_is_none_when_single_class() -> None:
    m = compute_metrics([0, 0], np.array([[0.9, 0.1], [0.8, 0.2]]), ["no", "yes"])
    assert m["roc_auc"] is None and m["pr_auc"] is None and m["accuracy"] == 1.0


def test_multiclass_metrics() -> None:
    y = [0, 1, 2, 2]
    p = np.eye(3)[[0, 1, 2, 1]] * 0.8 + 0.2 / 3
    m = compute_metrics(y, p, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(0.75)
    assert np.array(m["confusion_matrix"]).shape == (3, 3)
    assert m["roc_auc"] is not None


def test_class_weights_inverse_frequency() -> None:
    w = class_weights([0, 0, 0, 1], 3)
    assert torch.allclose(w, torch.tensor([4 / 6, 4 / 2, 1.0]))


def test_early_stopping() -> None:
    es = EarlyStopping(patience=2, min_delta=0.01)
    assert es.step(0.5) and not es.step(0.505) and not es.should_stop
    assert not es.step(0.4) and es.should_stop
    assert es.step(0.6) and not es.should_stop


def test_folds_keep_videos_together() -> None:
    labels = np.array([0, 1] * 12)
    groups = np.repeat(np.arange(6), 4)
    folds = make_folds(labels, groups, k=3, seed=0)
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        assert not set(groups[train_idx]) & set(groups[val_idx])
    assert sorted(np.concatenate([v for _, v in folds]).tolist()) == list(range(24))


def test_folds_fall_back_when_too_few_videos(caplog) -> None:
    folds = make_folds(np.array([0, 1] * 5), np.zeros(10, dtype=int), k=2, seed=0)
    assert len(folds) == 2 and "leak" in caplog.text


def test_folds_reduce_k_for_rare_class() -> None:
    folds = make_folds(np.array([0] * 10 + [1] * 2), np.arange(12), k=5, seed=0)
    assert len(folds) == 2
