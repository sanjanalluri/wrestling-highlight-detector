from pathlib import Path

import pytest

from wrestleai.db import Database


def test_video_upsert_and_lookup(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.db")
    vid = db.upsert_video(tmp_path / "a.mp4", 30.0, 300, 640, 480, 10.0)
    again = db.upsert_video(tmp_path / "a.mp4", 30.0, 330, 640, 480, 11.0)
    assert vid == again
    rec = db.get_video(vid)
    assert rec is not None and rec.frame_count == 330 and rec.read_path == rec.path
    assert db.find_video("a.mp4") == rec
    assert [v.id for v in db.list_videos()] == [vid]


def test_label_ranges_round_trip(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.db")
    vid = db.upsert_video(tmp_path / "a.mp4", 12.0, 120, 64, 64, 10.0)
    r2 = db.add_label_range(vid, 5.0, 7.5, "highlight", notes="double leg")
    r1 = db.add_label_range(vid, 0.0, 2.0, "takedown")
    ranges = db.list_label_ranges(vid)
    assert [r.id for r in ranges] == [r1, r2]  # ordered by start time
    assert ranges[1].notes == "double leg"
    assert db.delete_label_range(r1) and not db.delete_label_range(r1)
    with pytest.raises(ValueError):
        db.add_label_range(vid, 3.0, 3.0, "highlight")


def test_clip_windows_replace(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.db")
    vid = db.upsert_video(tmp_path / "a.mp4", 12.0, 120, 64, 64, 10.0)
    db.replace_clip_windows(vid, [(0, 0.0, 4.0, "non_highlight"), (1, 2.0, 6.0, "highlight")])
    db.replace_clip_windows(vid, [(0, 0.0, 5.0, "highlight")])
    windows = db.list_clip_windows([vid])
    assert len(windows) == 1 and windows[0].end_sec == 5.0
    assert db.get_clip_windows_by_ids([windows[0].id]) == windows
    assert db.list_clip_windows([]) == []


def test_data_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "x.db"
    vid = Database(path).upsert_video(tmp_path / "a.mp4", 12.0, 12, 8, 8, 1.0)
    assert Database(path).get_video(vid) is not None
