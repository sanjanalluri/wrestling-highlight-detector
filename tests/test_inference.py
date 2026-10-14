"""Inference: merging positive windows into highlight ranges."""

import pytest

from wrestleai.data.windowing import Window
from wrestleai.inference.predict import merge_positive_windows


def test_merge_positive_windows() -> None:
    windows = [Window(i, i * 1.0, i * 1.0 + 2.0) for i in range(8)]  # 2s windows, 1s stride
    probs = [0.1, 0.8, 0.9, 0.2, 0.1, 0.1, 0.7, 0.1]
    ranges = merge_positive_windows(windows, probs, threshold=0.5)
    assert [(r.start_sec, r.end_sec, r.num_windows) for r in ranges] == [(1, 4, 2), (6, 8, 1)]
    assert ranges[0].max_prob == pytest.approx(0.9)
    merged = merge_positive_windows(windows, probs, 0.5, max_gap=2.0)
    assert [(r.start_sec, r.end_sec) for r in merged] == [(1, 8)]
    assert merge_positive_windows(windows, probs, 0.5, min_duration=2.5)[0].end_sec == 4
    assert merge_positive_windows(windows, [0.0] * 8) == []
