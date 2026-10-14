from pathlib import Path
from typing import List

from wrestleai.db import Database
from wrestleai.labeling import RangeLabeler


def make_labeler(tmp_path: Path, answers: List[str]):
    db = Database(tmp_path / "x.db")
    vid = db.upsert_video(tmp_path / "match.mp4", 12.0, 2400, 64, 64, 200.0)
    out: List[str] = []
    labeler = RangeLabeler(db, db.get_video(vid), ["non_highlight", "highlight", "takedown"], "highlight",
                           labeled_by="tester", input_fn=lambda _: answers.pop(0), output_fn=out.append)
    return labeler, db, out


def test_add_list_delete_commands(tmp_path: Path) -> None:
    labeler, db, out = make_labeler(tmp_path, [])
    assert labeler.handle("0:00 0:05")  # 0:00 is valid (was rejected by the old labeler)
    assert labeler.handle("1:30 1:42 takedown # double leg")
    ranges = labeler.ranges()
    assert [(r.start_sec, r.end_sec, r.label) for r in ranges] == [(0, 5, "highlight"), (90, 102, "takedown")]
    assert ranges[1].notes == "double leg" and ranges[1].labeled_by == "tester"
    labeler.handle("list")
    assert any("takedown" in line for line in out)
    labeler.handle("delete 1")
    assert [r.label for r in labeler.ranges()] == ["takedown"]
    assert labeler.handle("done") is False


def test_rejects_bad_input(tmp_path: Path) -> None:
    labeler, _, out = make_labeler(tmp_path, [])
    for line in ["5 3", "1:00 2:00 pin", "0:10 300", "abc def", "delete 9", "1 2 3 4"]:
        labeler.handle(line)
    assert labeler.ranges() == [] and len(out) == 6


def test_overlap_requires_confirmation(tmp_path: Path) -> None:
    labeler, _, _ = make_labeler(tmp_path, ["n", "y"])
    labeler.handle("10 20")
    labeler.handle("15 25")  # answered "n"
    assert len(labeler.ranges()) == 1
    labeler.handle("15 25")  # answered "y"
    assert len(labeler.ranges()) == 2


def test_run_loop_ends_on_done_and_eof(tmp_path: Path) -> None:
    labeler, _, _ = make_labeler(tmp_path, ["0 4", "done"])
    labeler.run()
    assert len(labeler.ranges()) == 1

    def eof(_: str) -> str:
        raise EOFError

    labeler.input_fn = eof
    labeler.run()  # must not raise
