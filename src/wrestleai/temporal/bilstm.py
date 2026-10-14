"""Stage 4: bidirectional LSTM over the per-frame feature sequence."""

from __future__ import annotations

import torch
import torch.nn as nn

from wrestleai.config import TemporalConfig


class BiLSTMEncoder(nn.Module):
    """Encodes ``(B, T, D)`` feature sequences into a ``(B, out_dim)`` clip embedding.

    ``pooling="mean"`` averages the LSTM outputs over time; ``"last"`` concatenates the
    final forward state with the final backward state (the backward pass ends at t=0).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 384,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be 'mean' or 'last'")
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,  # PyTorch only applies it between layers
        )
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.out_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(f"expected (B, T, D) features, got {tuple(features.shape)}")
        outputs, (h_n, _) = self.lstm(features)  # outputs: (B, T, out_dim)
        if self.pooling == "mean":
            return outputs.mean(dim=1)
        if self.bidirectional:
            return torch.cat([h_n[-2], h_n[-1]], dim=-1)  # last layer: forward, backward
        return h_n[-1]


def build_temporal(cfg: TemporalConfig, input_dim: int) -> BiLSTMEncoder:
    return BiLSTMEncoder(input_dim, cfg.hidden_size, cfg.num_layers, cfg.dropout, cfg.bidirectional, cfg.pooling)
