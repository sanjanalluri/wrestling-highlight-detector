"""Stage 5 classifier head and the composed HighlightModel."""

import pytest
import torch

from wrestleai.models import ClassifierHead, build_model


@pytest.mark.parametrize("num_classes", [2, 5])
def test_classifier_output(num_classes: int) -> None:
    head = ClassifierHead(16, num_classes, hidden_dims=[8], dropout=0.0)
    logits = head(torch.randn(3, 16))
    assert logits.shape == (3, num_classes)
    assert torch.allclose(torch.softmax(logits, -1).sum(-1), torch.ones(3))


def test_full_model_forward_and_backward(cfg) -> None:
    model = build_model(cfg)
    clips = torch.randn(2, cfg.data.frames_per_window, 3, *cfg.data.resize)
    logits = model(clips)
    assert logits.shape == (2, cfg.labels.num_classes)
    logits.sum().backward()
    assert model.head.net[0].weight.grad is not None


def test_full_model_with_segmentation_multiclass(cfg) -> None:
    cfg.segmentation.enabled = True
    cfg.labels.classes = ["none", "takedown", "pin"]
    cfg.labels.positive_class = "takedown"
    model = build_model(cfg)
    assert model.segmenter is not None
    probs = model.predict_proba(torch.randn(2, 3, 3, 64, 64))
    assert probs.shape == (2, 3)


def test_model_rejects_wrong_rank(cfg) -> None:
    with pytest.raises(ValueError):
        build_model(cfg)(torch.randn(2, 3, 64, 64))
