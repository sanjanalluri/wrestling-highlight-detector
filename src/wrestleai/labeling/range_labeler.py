"""Interactive terminal tool for marking labeled time ranges on a video.

Ported from the original Groot terminal labeler. Commands::

    <start> <end> [label] [# notes]   add a range, e.g. "1:30 1:42 highlight # double leg"
    list                              show ranges for this video
    delete <n>                        delete range number n from the list
    play [start]                      open the video in the system player
    help                              show commands
    done                              finish (ranges are saved as you go)

Times accept ``SS``, ``MM:SS`` or ``HH:MM:SS`` (decimals allowed). The label defaults
to ``labels.positive_class`` and must be one of ``labels.classes``.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys
from typing import Callable, List, Optional, Sequence

from wrestleai.db.database import Database, LabelRangeRecord, VideoRecord
from wrestleai.utils.timefmt import format_time, parse_time

logger = logging.getLogger(__name__)

HELP = __doc__.split("Commands::", 1)[1].split("Times accept", 1)[0].rstrip()


class RangeLabeler:
    """Command interpreter for labeling one video. I/O is injectable for testing."""

    def __init__(
        self,
        db: Database,
        video: VideoRecord,
        classes: Sequence[str],
        default_label: str,
        labeled_by: str = "manual",
        input_fn: Callable[[str], str] = input,
        output_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.db = db
        self.video = video
        self.classes = list(classes)
        self.default_label = default_label
        self.labeled_by = labeled_by
        self.input_fn = input_fn
        self.output_fn = output_fn or (lambda msg: sys.stdout.write(msg + "\n"))

    # -------------------------------------------------------------- helpers
    def say(self, message: str) -> None:
        """Write a user-facing line (UI output, distinct from logging)."""
        self.output_fn(message)

    def ranges(self) -> List[LabelRangeRecord]:
        return self.db.list_label_ranges(self.video.id)

    def show_ranges(self) -> None:
        ranges = self.ranges()
        if not ranges:
            self.say("No ranges labeled yet.")
            return
        for i, r in enumerate(ranges, 1):
            note = f"  # {r.notes}" if r.notes else ""
            self.say(f"{i:3d}. {format_time(r.start_sec)} -> {format_time(r.end_sec)}  {r.label}{note}")

    # ------------------------------------------------------------- commands
    def add(self, start_sec: float, end_sec: float, label: str, notes: Optional[str] = None,
            confirm_overlap: bool = True) -> Optional[int]:
        """Validate and store a range; return its id or ``None`` if rejected."""
        if label not in self.classes:
            self.say(f"Unknown label {label!r}. Choose from: {', '.join(self.classes)}")
            return None
        if end_sec <= start_sec:
            self.say("End time must be after start time.")
            return None
        duration = self.video.duration_seconds
        if duration > 0 and end_sec > duration + 1e-6:
            self.say(f"End time exceeds video duration ({format_time(duration)}).")
            return None
        overlaps = [r for r in self.ranges() if start_sec < r.end_sec and end_sec > r.start_sec]
        if overlaps and confirm_overlap:
            for r in overlaps:
                self.say(f"Warning: overlaps {format_time(r.start_sec)} -> {format_time(r.end_sec)} ({r.label})")
            if self.input_fn("Add anyway? (y/n): ").strip().lower() != "y":
                return None
        range_id = self.db.add_label_range(self.video.id, start_sec, end_sec, label, self.labeled_by, notes)
        logger.info("Added %s range %.2f-%.2fs to %s (id=%d)", label, start_sec, end_sec, self.video.name, range_id)
        self.say(f"Added {label}: {format_time(start_sec)} -> {format_time(end_sec)}")
        return range_id

    def handle(self, line: str) -> bool:
        """Execute one command line. Returns ``False`` when the session should end."""
        notes: Optional[str] = None
        if "#" in line:
            line, notes = line.split("#", 1)
            notes = notes.strip() or None
        parts = line.strip().split()
        if not parts:
            return True
        cmd = parts[0].lower()
        if cmd in {"done", "quit", "exit", "q"}:
            return False
        if cmd == "help":
            self.say(HELP)
        elif cmd == "list":
            self.show_ranges()
        elif cmd == "delete":
            self._delete(parts[1:])
        elif cmd == "play":
            start = parts[1] if len(parts) > 1 else "0"
            try:
                play_video(self.video.path, parse_time(start))
            except ValueError as exc:
                self.say(str(exc))
        else:
            self._add_from_parts(parts, notes)
        return True

    def _delete(self, args: List[str]) -> None:
        ranges = self.ranges()
        try:
            n = int(args[0])
            if n < 1:
                raise IndexError
            target = ranges[n - 1]
        except (IndexError, ValueError):
            self.say(f"Usage: delete <n> with n between 1 and {len(ranges)}")
            return
        self.db.delete_label_range(target.id)
        logger.info("Deleted range id=%d from %s", target.id, self.video.name)
        self.say(f"Deleted {format_time(target.start_sec)} -> {format_time(target.end_sec)} ({target.label})")

    def _add_from_parts(self, parts: List[str], notes: Optional[str]) -> None:
        if len(parts) not in (2, 3):
            self.say("Usage: <start> <end> [label]   (type 'help' for all commands)")
            return
        try:
            start, end = parse_time(parts[0]), parse_time(parts[1])
        except ValueError as exc:
            self.say(str(exc))
            return
        label = parts[2] if len(parts) == 3 else self.default_label
        self.add(start, end, label, notes)

    def run(self) -> None:
        """Interactive loop until ``done`` or EOF."""
        self.say(f"Labeling {self.video.name} ({format_time(self.video.duration_seconds)})")
        self.say(f"Labels: {', '.join(self.classes)} (default: {self.default_label})")
        self.say(HELP)
        self.show_ranges()
        while True:
            try:
                line = self.input_fn(f"[{self.video.name}] > ")
            except (EOFError, KeyboardInterrupt):
                self.say("")
                break
            if not self.handle(line):
                break
        self.say(f"Saved {len(self.ranges())} ranges for {self.video.name}.")


def play_video(path: str, start_sec: float = 0.0) -> None:
    """Best-effort: open ``path`` in a system player (VLC seeks to ``start_sec`` if installed)."""
    vlc = shutil.which("vlc")
    if vlc:
        cmd = [vlc, path, f"--start-time={start_sec:.1f}"]
    elif platform.system() == "Darwin":
        cmd = ["open", path]
    elif platform.system() == "Windows":
        cmd = ["cmd", "/c", "start", "", path]
    else:
        cmd = ["xdg-open", path]
    try:
        subprocess.Popen(cmd)  # noqa: S603 - fixed argv, user-selected file
        logger.info("Opened %s (seek manually to %s if needed)", path, format_time(start_sec))
    except OSError as exc:
        logger.warning("Could not launch a video player: %s", exc)
