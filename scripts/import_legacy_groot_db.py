#!/usr/bin/env python3
"""Copy videos and labeled ranges from the original Groot database into the new schema.

Reads ``groot_videos`` / ``groot_highlight_clips`` from the legacy SQLite file (opened
read-only; it is never modified) and writes them into ``paths.db_path``. Relative video
paths in the legacy DB are resolved against ``--legacy-root`` (the directory the old
scripts ran from, ``groot/`` by default). Only videos that have labeled clips are copied.

Example:
    python scripts/import_legacy_groot_db.py --legacy-db database/wrestling_pipeline.db
    wrestleai extract --labeled-only
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

from wrestleai.config import load_config
from wrestleai.data.video import probe_video
from wrestleai.db.database import Database
from wrestleai.utils.logging import setup_logging

logger = logging.getLogger("import_legacy_groot_db")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy-db", type=Path, default=Path("database/wrestling_pipeline.db"))
    parser.add_argument("--legacy-root", type=Path, default=Path("groot"),
                        help="directory relative legacy video paths are resolved against")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    setup_logging("INFO")
    cfg = load_config(args.config if args.config.exists() else None, args.overrides)
    if not args.legacy_db.exists():
        logger.error("Legacy database not found: %s", args.legacy_db)
        return 1
    legacy = sqlite3.connect(f"file:{args.legacy_db.resolve()}?mode=ro", uri=True)
    legacy.row_factory = sqlite3.Row
    rows = legacy.execute(
        """SELECT v.id, v.video_path, v.video_name, c.start_time_seconds, c.end_time_seconds,
                  c.is_highlight, c.labeled_by, c.notes
           FROM groot_highlight_clips c JOIN groot_videos v ON v.id = c.video_id
           ORDER BY v.id, c.start_time_seconds"""
    ).fetchall()
    legacy.close()
    db = Database(cfg.paths.db_path)
    new_ids: dict[int, int] = {}
    imported = skipped = 0
    for row in rows:
        if row["id"] not in new_ids:
            path = Path(row["video_path"])
            if not path.is_absolute():
                path = args.legacy_root / path
            if not path.exists():
                logger.warning("Skipping %s: file not found at %s", row["video_name"], path)
                new_ids[row["id"]] = -1
            else:
                info = probe_video(path)
                new_ids[row["id"]] = db.upsert_video(path, info.fps, info.frame_count, info.width,
                                                     info.height, info.duration_seconds)
                existing = db.list_label_ranges(new_ids[row["id"]])
                for r in existing:
                    if r.labeled_by.startswith("legacy:"):
                        db.delete_label_range(r.id)  # make re-running idempotent
        video_id = new_ids[row["id"]]
        if video_id < 0:
            skipped += 1
            continue
        label = cfg.labels.positive_class if row["is_highlight"] else cfg.labels.default_class
        db.add_label_range(video_id, row["start_time_seconds"], row["end_time_seconds"], label,
                           labeled_by=f"legacy:{row['labeled_by'] or 'manual'}", notes=row["notes"])
        imported += 1
    logger.info("Imported %d labeled ranges for %d videos into %s (%d skipped)",
                imported, sum(1 for v in new_ids.values() if v >= 0), cfg.paths.db_path, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
