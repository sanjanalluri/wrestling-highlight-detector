"""Stage 2: DeepLabV3+ segmentation and masking (random weights, no downloads)."""

import pytest
import torch

from wrestleai.segmentation import SilhouetteSegmenter, apply_masks


@pytest.fixture(scope="module")
def segmenter() -> SilhouetteSegmenter:
    return SilhouetteSegmenter("resnet18", encoder_weights=None, input_size=(64, 64))


def test_segmenter_mask_shape_and_range(segmenter: SilhouetteSegmenter) -> None:
    masks = segmenter(torch.randn(3, 3, 40, 56))  # resized to 64x64 internally, then back
    assert masks.shape == (3, 1, 40, 56)
    assert masks.min() >= 0 and masks.max() <= 1


def test_frozen_segmenter_has_no_trainable_params(segmenter: SilhouetteSegmenter) -> None:
    segmenter.train()
    assert not any(p.requires_grad for p in segmenter.parameters())
    assert not segmenter.net.training


def test_segmenter_rejects_bad_input_size() -> None:
    with pytest.raises(ValueError):
        SilhouetteSegmenter("resnet18", encoder_weights=None, input_size=(72, 72))


def test_apply_masks_hard_and_soft() -> None:
    frames = torch.ones(2, 3, 4, 4)
    masks = torch.zeros(2, 1, 4, 4)
    masks[:, :, :2] = 0.9
    masks[:, :, 2:] = 0.2
    hard = apply_masks(frames, masks, threshold=0.5)
    assert torch.all(hard[:, :, :2] == 1) and torch.all(hard[:, :, 2:] == 0)
    soft = apply_masks(frames, masks, soft=True)
    assert torch.allclose(soft[:, :, 2:], torch.full_like(soft[:, :, 2:], 0.2))
    with pytest.raises(ValueError):
        apply_masks(frames, torch.zeros(1, 1, 4, 4))
