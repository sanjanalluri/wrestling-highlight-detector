"""Stage 2: DeepLabV3+ wrestler silhouette segmentation and frame masking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wrestleai.config import SegmentationConfig

logger = logging.getLogger(__name__)


class SilhouetteSegmenter(nn.Module):
    """Binary person/background segmenter built on ``segmentation_models_pytorch``.

    Frames are resized to ``input_size`` (must be divisible by 16 for DeepLabV3+),
    segmented, and the mask is resized back to the frame resolution.
    """

    def __init__(
        self,
        encoder: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        input_size: Tuple[int, int] = (512, 512),
        checkpoint: Optional[str | Path] = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp  # local import keeps module import cheap

        if input_size[0] % 16 or input_size[1] % 16:
            raise ValueError(f"segmentation.input_size {input_size} must be divisible by 16")
        self.input_size = tuple(input_size)
        self.net = smp.DeepLabV3Plus(
            encoder_name=encoder, encoder_weights=encoder_weights, classes=1, activation=None
        )
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            state = state.get("state_dict", state) if isinstance(state, dict) else state
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            logger.info("Loaded segmentation weights from %s (missing=%d, unexpected=%d)",
                        checkpoint, len(missing), len(unexpected))
        else:
            logger.warning(
                "DeepLabV3+ has no fine-tuned checkpoint (segmentation.checkpoint); the decoder is "
                "randomly initialised, so masks are not meaningful until it is trained on person masks."
            )
        self.frozen = freeze
        if freeze:
            for p in self.net.parameters():
                p.requires_grad_(False)
            self.net.eval()

    def train(self, mode: bool = True) -> "SilhouetteSegmenter":
        """Keep a frozen segmenter in eval mode (stable BatchNorm statistics)."""
        super().train(mode)
        if self.frozen:
            self.net.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Return foreground probabilities ``(N, 1, H, W)`` for frames ``(N, 3, H, W)``."""
        h, w = frames.shape[-2:]
        x = frames
        if (h, w) != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        with torch.set_grad_enabled(not self.frozen and torch.is_grad_enabled()):
            logits = self.net(x)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)


def apply_masks(
    frames: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5, soft: bool = False
) -> torch.Tensor:
    """Zero out background pixels.

    Args:
        frames: ``(N, 3, H, W)`` normalised frames (0 == the dataset mean colour).
        masks: ``(N, 1, H, W)`` foreground probabilities in ``[0, 1]``.
        threshold: cut-off for hard masks.
        soft: multiply by the probabilities instead of a binary mask.
    """
    if masks.shape[0] != frames.shape[0] or masks.shape[-2:] != frames.shape[-2:]:
        raise ValueError(f"mask shape {tuple(masks.shape)} does not match frames {tuple(frames.shape)}")
    weight = masks if soft else (masks >= threshold).to(frames.dtype)
    return frames * weight


def build_segmenter(cfg: SegmentationConfig) -> Optional[SilhouetteSegmenter]:
    """Return a segmenter when ``cfg.enabled``; otherwise ``None``."""
    if not cfg.enabled:
        return None
    return SilhouetteSegmenter(cfg.encoder, cfg.encoder_weights, tuple(cfg.input_size),  # type: ignore[arg-type]
                               cfg.checkpoint, cfg.freeze)
