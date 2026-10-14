"""Stage 3: per-frame CNN feature extraction (EfficientNet-B3 by default, via timm)."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from wrestleai.config import FeaturesConfig

logger = logging.getLogger(__name__)


class FrameFeatureExtractor(nn.Module):
    """Wraps a timm backbone with its classifier removed; outputs one vector per frame.

    With ``freeze=True`` the backbone weights are not trained and BatchNorm stays in
    eval mode, so it behaves as a fixed pretrained feature extractor.
    """

    def __init__(self, backbone: str = "efficientnet_b3", pretrained: bool = True, freeze: bool = True) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.out_dim: int = int(self.backbone.num_features)
        self.frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
        logger.info("Feature extractor %s (pretrained=%s, frozen=%s) -> %d-d features",
                    backbone, pretrained, freeze, self.out_dim)

    def train(self, mode: bool = True) -> "FrameFeatureExtractor":
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Map ``(N, 3, H, W)`` frames to ``(N, out_dim)`` features."""
        if self.frozen:
            with torch.no_grad():
                return self.backbone(frames)
        return self.backbone(frames)


def build_feature_extractor(cfg: FeaturesConfig) -> FrameFeatureExtractor:
    return FrameFeatureExtractor(cfg.backbone, cfg.pretrained, cfg.freeze)
