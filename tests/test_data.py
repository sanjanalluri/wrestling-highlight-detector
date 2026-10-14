from pathlib import Path

import numpy as np
import pytest
import torch

from wrestleai.data.dataset import ClipWindowDataset, WindowSample, frames_to_tensor, samples_from_db
from wrestleai.data.dedup import deduplicate
from wrestleai.data.extract import extract
from wrestleai.data.labels import LabelRange, assign_window_label, load_label_file
from wrestleai.data.video import extract_frames, probe_video, read_window
from wrestleai.data.windowing import Window, compute_windows
from wrestleai.db import Database
from wrestleai.utils.timefmt import format_time, parse_time


# ------------------------------------------------------------------ windowing
def test_windows_cover_video_with_stride() -> None:
    w = compute_windows(10.0, 4.0, 2.0)
    assert [(x.start_sec, x.end_sec) for x in w] == [(0, 4), (2, 6), (4, 8), (6, 10)]
    assert [x.index for x in w] == [0, 1, 2, 3]


def test_tail_window_added_when_stride_leaves_gap() -> None:
    w = compute_windows(9.0, 4.0, 2.5, min_window_seconds=1.0)
    assert w[-1].end_sec == 9.0 and w[-1].duration == pytest.approx(4.0)


def test_short_video_gets_single_short_window_or_none() -> None:
    assert [(x.start_sec, x.end_sec) for x in compute_windows(2.0, 5.0, 2.5, 1.0)] == [(0.0, 2.0)]
    assert compute_windows(0.5, 5.0, 2.5, 1.0) == []
    assert compute_windows(0.0, 5.0, 2.5) == []


# -------------------------------------------------------------- label mapping
def test_window_label_uses_overlap_threshold() -> None:
    ranges = [LabelRange(3.0, 6.0, "highlight")]
    assert assign_window_label(Window(0, 0.0, 4.0), ranges, "none", 0.5) == "none"  # 25%
    assert assign_window_label(Window(1, 2.0, 6.0), ranges, "none", 0.5) == "highlight"  # 75%
    assert assign_window_label(Window(2, 5.0, 9.0), ranges, "none", 0.5) == "none"  # 25%
    assert assign_window_label(Window(2, 5.0, 9.0), ranges, "none", 0.25) == "highlight"


def test_adjacent_ranges_of_same_label_are_summed() -> None:
    ranges = [LabelRange(0.0, 1.5, "highlight"), LabelRange(1.5, 3.0, "highlight")]
    assert assign_window_label(Window(0, 0.0, 4.0), ranges, "none", 0.7) == "highlight"


def test_multiclass_picks_label_with_most_coverage() -> None:
    ranges = [LabelRange(0.0, 1.0, "takedown"), LabelRange(1.0, 4.0, "pin")]
    assert assign_window_label(Window(0, 0.0, 4.0), ranges, "none", 0.5) == "pin"


def test_invalid_range_rejected() -> None:
    with pytest.raises(ValueError):
        LabelRange(5.0, 5.0, "highlight")


# ---------------------------------------------------------------- time parsing
@pytest.mark.parametrize(
    "text,seconds", [("0:00", 0.0), ("90", 90.0), ("1:30", 90.0), ("1:02:03", 3723.0), ("0:01.5", 1.5)]
)
def test_parse_time(text: str, seconds: float) -> None:
    assert parse_time(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "a:b", "1:75", "-3", "1::2"])
def test_parse_time_rejects_garbage(text: str) -> None:
    with pytest.raises(ValueError):
        parse_time(text)


def test_format_time() -> None:
    assert format_time(90.25) == "1:30.2" and format_time(3725) == "1:02:05.0"


# ------------------------------------------------------------- video + dataset
def test_probe_and_extract_resample(synthetic_videos) -> None:
    video, _ = synthetic_videos[0]
    info = probe_video(video)
    assert info.fps == pytest.approx(12.0) and info.duration_seconds == pytest.approx(16.0, abs=0.1)
    frames, stamps = extract_frames(video, 6.0, 2.0, 4.0, resize=(32, 48))
    assert frames.shape == (12, 32, 48, 3) and frames.dtype == np.uint8
    assert stamps[0] == pytest.approx(2.0) and np.all(np.diff(stamps) > 0)


def test_read_window_returns_fixed_length(synthetic_videos) -> None:
    video, _ = synthetic_videos[0]
    assert read_window(video, 14.5, 16.0, 6.0, 8, (16, 16)).shape == (8, 16, 16, 3)


def test_frames_to_tensor_normalises() -> None:
    frames = np.full((2, 4, 4, 3), 255, dtype=np.uint8)
    t = frames_to_tensor(frames, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    assert t.shape == (2, 3, 4, 4) and torch.allclose(t, torch.ones_like(t))


def test_dataset_shapes(cfg, synthetic_videos) -> None:
    video, _ = synthetic_videos[0]
    ds = ClipWindowDataset([WindowSample(str(video), 0.0, 2.0, 1)], cfg)
    frames, label = ds[0]
    h, w = cfg.data.resize
    assert frames.shape == (cfg.data.frames_per_window, 3, h, w) and label == 1


def test_extract_builds_labeled_windows(cfg, synthetic_videos) -> None:
    db = Database(cfg.paths.db_path)
    video, labels_csv = synthetic_videos[0]
    (rec,) = extract(db, cfg, [video])
    for r in load_label_file(labels_csv):
        db.add_label_range(rec.id, r.start_sec, r.end_sec, r.label)
    extract(db, cfg)  # re-window after labeling
    windows = db.list_clip_windows([rec.id])
    assert len(windows) == len(compute_windows(rec.duration_seconds, 2.0, 1.0, 1.0))
    assert {w.label for w in windows} == {"non_highlight", "highlight"}
    samples = samples_from_db(db, cfg)
    assert len(samples) == len(windows) and {s.label for s in samples} == {0, 1}


def test_extract_with_conversion(cfg, synthetic_videos) -> None:
    db = Database(cfg.paths.db_path)
    (rec,) = extract(db, cfg, [synthetic_videos[1][0]], convert=True)
    assert rec.processed_path is not None and Path(rec.processed_path).exists()
    info = probe_video(rec.processed_path)
    assert info.fps == pytest.approx(cfg.data.target_fps)
    assert (info.height, info.width) == tuple(cfg.data.resize)


def test_dedup_drops_repeated_frames(synthetic_videos) -> None:
    frames, _ = extract_frames(synthetic_videos[0][0], 24.0, 0.0, 1.0)  # 2x upsampled
    keep = deduplicate(frames)
    assert len(frames) == 24 and len(keep) <= 13
