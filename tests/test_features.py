"""Stage 3: per-frame CNN feature extractor (weights=None, no downloads)."""

import torch

from wrestleai.features.cnn import FrameFeatureExtractor


def test_efficientnet_b3_feature_dim() -> None:
    fx = FrameFeatureExtractor("efficientnet_b3", pretrained=False, freeze=True)
    out = fx(torch.randn(2, 3, 64, 64))
    assert fx.out_dim == 1536 and out.shape == (2, 1536)


def test_frozen_backbone_has_no_grad() -> None:
    fx = FrameFeatureExtractor("efficientnet_b0", pretrained=False, freeze=True)
    fx.train()
    assert not fx.backbone.training
    assert not fx(torch.randn(1, 3, 32, 32)).requires_grad
