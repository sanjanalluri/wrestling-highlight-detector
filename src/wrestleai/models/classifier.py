"""Stage 5: dense classification head (binary highlight or multi-class technique)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """MLP ``in_dim -> hidden_dims... -> num_classes`` returning raw logits.

    Binary classification uses two logits with softmax/cross-entropy, so the same head
    and loss handle any number of classes.
    """

    def __init__(self, in_dim: int, num_classes: int, hidden_dims: Sequence[int] = (512, 128), dropout: float = 0.3) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        layers = []
        prev = in_dim
        for width in hidden_dims:
            layers += [nn.Linear(prev, width), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
