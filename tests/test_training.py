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


def test_checkpoint_omits_frozen_pretrained_backbone(cfg, tmp_path, monkeypatch) -> None:
    import timm

    from wrestleai.models import build_model
    from wrestleai.training.checkpoint import load_checkpoint, save_checkpoint

    real_create = timm.create_model

    def fake_pretrained(name, pretrained=False, **kwargs):
        # Stand-in for downloading weights: deterministic "pretrained" init, offline.
        torch.manual_seed(1234)
        return real_create(name, pretrained=False, **kwargs)

    monkeypatch.setattr(timm, "create_model", fake_pretrained)
    cfg.features.freeze, cfg.features.pretrained = True, True
    model = build_model(cfg).eval()
    small, full = tmp_path / "small.pt", tmp_path / "full.pt"
    save_checkpoint(small, model, cfg)
    torch.save({"model_state": model.state_dict(), "config": cfg.to_dict()}, full)

    ckpt = torch.load(small)
    assert ckpt["excluded_prefixes"] == ["features."]
    assert not any(k.startswith("features.") for k in ckpt["model_state"])
    assert small.stat().st_size < full.stat().st_size / 2

    clips = torch.randn(2, cfg.data.frames_per_window, 3, *cfg.data.resize)
    with torch.no_grad():
        expected = model(clips)
        for path in (small, full):  # new compact format and older full checkpoints both load
            loaded, _, _ = load_checkpoint(path)
            assert torch.allclose(loaded(clips), expected, atol=1e-5)


def test_trained_backbone_is_kept_in_checkpoint(cfg, tmp_path) -> None:
    from wrestleai.models import build_model
    from wrestleai.training.checkpoint import load_checkpoint, save_checkpoint

    model = build_model(cfg).eval()  # test config: backbone trained, not pretrained
    save_checkpoint(tmp_path / "c.pt", model, cfg)
    assert torch.load(tmp_path / "c.pt")["excluded_prefixes"] == []
    clips = torch.randn(1, cfg.data.frames_per_window, 3, *cfg.data.resize)
    with torch.no_grad():
        assert torch.allclose(load_checkpoint(tmp_path / "c.pt")[0](clips), model(clips), atol=1e-5)


def test_checkpoint_from_before_motion_features_loads(cfg, tmp_path) -> None:
    from wrestleai.models import build_model
    from wrestleai.training.checkpoint import load_checkpoint

    model = build_model(cfg).eval()
    old_config = cfg.to_dict()
    del old_config["features"]["motion"]  # as written by versions without motion features
    torch.save({"model_state": model.state_dict(), "config": old_config}, tmp_path / "old.pt")
    loaded, loaded_cfg, _ = load_checkpoint(tmp_path / "old.pt")
    assert loaded_cfg.features.motion is False
    clips = torch.randn(1, cfg.data.frames_per_window, 3, *cfg.data.resize)
    with torch.no_grad():
        assert torch.allclose(loaded(clips), model(clips), atol=1e-5)
