"""SQLite storage for videos, labeled time ranges and clip-window metadata.

The schema follows the original Groot tables (``groot_videos`` /
``groot_highlight_clips``) with two changes: ranges carry a free-text ``label`` so
multi-class technique labels work, and a ``clip_windows`` table records the fixed
windows each video is sliced into together with their mapped label.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    fps REAL,
    frame_count INTEGER,
    width INTEGER,
    height INTEGER,
    duration_seconds REAL,
    processed_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS label_ranges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    label TEXT NOT NULL,
    labeled_by TEXT DEFAULT 'manual',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (end_sec > start_sec)
);

CREATE TABLE IF NOT EXISTS clip_windows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    window_index INTEGER NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    label TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (video_id, window_index)
);

CREATE INDEX IF NOT EXISTS idx_label_ranges_video ON label_ranges(video_id);
CREATE INDEX IF NOT EXISTS idx_clip_windows_video ON clip_windows(video_id);
"""


@dataclass(frozen=True)
class VideoRecord:
    """A registered source video."""

    id: int
    path: str
    name: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float
    processed_path: Optional[str] = None

    @property
    def read_path(self) -> str:
        """Path frames should be decoded from (the resampled copy when one exists)."""
        return self.processed_path or self.path


@dataclass(frozen=True)
class LabelRangeRecord:
    """A labeled time range within a video."""

    id: int
    video_id: int
    start_sec: float
    end_sec: float
    label: str
    labeled_by: str = "manual"
    notes: Optional[str] = None


@dataclass(frozen=True)
class ClipWindowRecord:
    """A fixed-length window of a video and the label mapped onto it."""

    id: int
    video_id: int
    window_index: int
    start_sec: float
    end_sec: float
    label: str


class Database:
    """Thin repository over a SQLite file. Each method opens a short-lived connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection that commits on success and rolls back on error."""
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create tables if they do not exist."""
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ videos
    def upsert_video(
        self,
        path: str | Path,
        fps: float,
        frame_count: int,
        width: int,
        height: int,
        duration_seconds: float,
        name: Optional[str] = None,
        processed_path: Optional[str | Path] = None,
    ) -> int:
        """Insert or update a video keyed by its absolute path; return its id."""
        abs_path = str(Path(path).resolve())
        processed = str(Path(processed_path).resolve()) if processed_path else None
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO videos (path, name, fps, frame_count, width, height,
                                    duration_seconds, processed_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    name = excluded.name, fps = excluded.fps,
                    frame_count = excluded.frame_count, width = excluded.width,
                    height = excluded.height, duration_seconds = excluded.duration_seconds,
                    processed_path = COALESCE(excluded.processed_path, videos.processed_path),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (abs_path, name or Path(path).name, fps, frame_count, width, height,
                 duration_seconds, processed),
            )
            row = conn.execute("SELECT id FROM videos WHERE path = ?", (abs_path,)).fetchone()
        return int(row["id"])

    def get_video(self, video_id: int) -> Optional[VideoRecord]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        return _video(row) if row else None

    def find_video(self, path_or_name: str | Path) -> Optional[VideoRecord]:
        """Look a video up by absolute path, then by file name."""
        key = str(path_or_name)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE path = ?", (str(Path(key).resolve()),)
            ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT * FROM videos WHERE name = ? ORDER BY id LIMIT 1", (Path(key).name,)
                ).fetchone()
        return _video(row) if row else None

    def list_videos(self) -> List[VideoRecord]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM videos ORDER BY name").fetchall()
        return [_video(r) for r in rows]

    # ------------------------------------------------------------ label ranges
    def add_label_range(
        self,
        video_id: int,
        start_sec: float,
        end_sec: float,
        label: str,
        labeled_by: str = "manual",
        notes: Optional[str] = None,
    ) -> int:
        """Store a labeled range and return its id."""
        if end_sec <= start_sec:
            raise ValueError(f"end_sec ({end_sec}) must be greater than start_sec ({start_sec})")
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO label_ranges (video_id, start_sec, end_sec, label, labeled_by, notes)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (video_id, float(start_sec), float(end_sec), label, labeled_by, notes),
            )
            return int(cur.lastrowid)

    def list_label_ranges(self, video_id: int) -> List[LabelRangeRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM label_ranges WHERE video_id = ? ORDER BY start_sec, id", (video_id,)
            ).fetchall()
        return [_range(r) for r in rows]

    def delete_label_range(self, range_id: int) -> bool:
        """Delete one range; return whether a row was removed."""
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM label_ranges WHERE id = ?", (range_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------ clip windows
    def replace_clip_windows(
        self, video_id: int, windows: Sequence[tuple[int, float, float, str]]
    ) -> int:
        """Replace all windows of a video with ``(window_index, start, end, label)`` rows."""
        with self.connect() as conn:
            conn.execute("DELETE FROM clip_windows WHERE video_id = ?", (video_id,))
            conn.executemany(
                "INSERT INTO clip_windows (video_id, window_index, start_sec, end_sec, label)"
                " VALUES (?, ?, ?, ?, ?)",
                [(video_id, int(i), float(s), float(e), lab) for i, s, e, lab in windows],
            )
        return len(windows)

    def list_clip_windows(
        self, video_ids: Optional[Sequence[int]] = None
    ) -> List[ClipWindowRecord]:
        """Return windows (optionally restricted to ``video_ids``) ordered by video and time."""
        query = "SELECT * FROM clip_windows"
        params: tuple = ()
        if video_ids is not None:
            if not video_ids:
                return []
            query += f" WHERE video_id IN ({','.join('?' * len(video_ids))})"
            params = tuple(int(v) for v in video_ids)
        query += " ORDER BY video_id, window_index"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_window(r) for r in rows]

    def get_clip_windows_by_ids(self, ids: Sequence[int]) -> List[ClipWindowRecord]:
        """Fetch windows by primary key, preserving the order of ``ids``."""
        if not ids:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM clip_windows WHERE id IN ({','.join('?' * len(ids))})",
                tuple(int(i) for i in ids),
            ).fetchall()
        by_id = {int(r["id"]): _window(r) for r in rows}
        missing = [i for i in ids if int(i) not in by_id]
        if missing:
            raise KeyError(f"clip_windows ids not found: {missing[:5]}...")
        return [by_id[int(i)] for i in ids]


def _video(row: sqlite3.Row) -> VideoRecord:
    return VideoRecord(
        id=int(row["id"]), path=row["path"], name=row["name"], fps=float(row["fps"] or 0.0),
        frame_count=int(row["frame_count"] or 0), width=int(row["width"] or 0),
        height=int(row["height"] or 0), duration_seconds=float(row["duration_seconds"] or 0.0),
        processed_path=row["processed_path"],
    )


def _range(row: sqlite3.Row) -> LabelRangeRecord:
    return LabelRangeRecord(
        id=int(row["id"]), video_id=int(row["video_id"]), start_sec=float(row["start_sec"]),
        end_sec=float(row["end_sec"]), label=row["label"], labeled_by=row["labeled_by"],
        notes=row["notes"],
    )


def _window(row: sqlite3.Row) -> ClipWindowRecord:
    return ClipWindowRecord(
        id=int(row["id"]), video_id=int(row["video_id"]), window_index=int(row["window_index"]),
        start_sec=float(row["start_sec"]), end_sec=float(row["end_sec"]), label=row["label"],
    )
