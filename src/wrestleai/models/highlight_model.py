"""The full Groot model: [segmentation] -> CNN features -> Bi-LSTM -> classifier."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from wrestleai.config import Config
from wrestleai.features.cnn import FrameFeatureExtractor, build_feature_extractor
from wrestleai.models.classifier import ClassifierHead
from wrestleai.segmentation.deeplab import SilhouetteSegmenter, apply_masks, build_segmenter
from wrestleai.temporal.bilstm import BiLSTMEncoder, build_temporal

logger = logging.getLogger(__name__)


class HighlightModel(nn.Module):
    """Classifies a clip ``(B, T, 3, H, W)`` into ``num_classes`` logits.

    Segmentation is an optional preprocessing step: when a segmenter is attached, every
    frame is masked to its wrestler silhouettes before feature extraction.
    """

    def __init__(
        self,
        features: FrameFeatureExtractor,
        temporal: BiLSTMEncoder,
        head: ClassifierHead,
        segmenter: Optional[SilhouetteSegmenter] = None,
        mask_threshold: float = 0.5,
        soft_mask: bool = False,
    ) -> None:
        super().__init__()
        self.segmenter = segmenter
        self.features = features
        self.temporal = temporal
        self.head = head
        self.mask_threshold = mask_threshold
        self.soft_mask = soft_mask

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply silhouette masks to flattened ``(N, 3, H, W)`` frames if segmentation is on."""
        if self.segmenter is None:
            return frames
        masks = self.segmenter(frames)
        return apply_masks(frames, masks, self.mask_threshold, self.soft_mask)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.dim() != 5:
            raise ValueError(f"expected (B, T, 3, H, W) clips, got {tuple(clips.shape)}")
        b, t = clips.shape[:2]
        frames = self.preprocess(clips.flatten(0, 1))
        feats = self.features(frames).view(b, t, -1)
        return self.head(self.temporal(feats))

    @torch.no_grad()
    def predict_proba(self, clips: torch.Tensor) -> torch.Tensor:
        """Softmax class probabilities ``(B, num_classes)``."""
        return torch.softmax(self.forward(clips), dim=-1)


def build_model(cfg: Config) -> HighlightModel:
    """Assemble a :class:`HighlightModel` from config."""
    segmenter = build_segmenter(cfg.segmentation)
    features = build_feature_extractor(cfg.features)
    temporal = build_temporal(cfg.temporal, features.out_dim)
    head = ClassifierHead(temporal.out_dim, cfg.labels.num_classes, cfg.classifier.hidden_dims, cfg.classifier.dropout)
    model = HighlightModel(features, temporal, head, segmenter, cfg.segmentation.threshold, cfg.segmentation.soft_mask)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Built HighlightModel: %d/%d trainable parameters, %d classes, segmentation=%s",
                trainable, total, cfg.labels.num_classes, segmenter is not None)
    return model
