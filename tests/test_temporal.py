"""Stage 4: Bi-LSTM temporal encoder."""

import pytest
import torch

from wrestleai.temporal import BiLSTMEncoder


@pytest.mark.parametrize("pooling", ["mean", "last"])
@pytest.mark.parametrize("bidirectional", [True, False])
def test_bilstm_output_shape(pooling: str, bidirectional: bool) -> None:
    enc = BiLSTMEncoder(32, hidden_size=8, num_layers=2, dropout=0.1, bidirectional=bidirectional, pooling=pooling)
    out = enc(torch.randn(4, 6, 32))
    assert out.shape == (4, 16 if bidirectional else 8) and enc.out_dim == out.shape[1]


def test_bilstm_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError):
        BiLSTMEncoder(8, 4)(torch.randn(4, 8))
