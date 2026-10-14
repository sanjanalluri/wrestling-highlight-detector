"""The ``extract`` stage: register videos, optionally resample them, and build labeled windows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

from wrestleai.config import Config
from wrestleai.data.labels import LabelRange, label_windows, load_label_file
from wrestleai.data.video import convert_video, find_videos, probe_video
from wrestleai.data.windowing import compute_windows
from wrestleai.db.database import Database, VideoRecord

logger = logging.getLogger(__name__)


def register_video(
    db: Database, path: str | Path, cfg: Config, convert: bool = False
) -> VideoRecord:
    """Probe ``path`` and store it; with ``convert`` also write a ``target_fps`` copy.

    The converted copy (resized to ``data.resize``) is what windows are decoded from,
    which keeps per-epoch decoding cheap for long, high-resolution match footage.
    """
    info = probe_video(path)
    processed: Optional[Path] = None
    if convert:
        processed = Path(cfg.paths.processed_dir) / f"{Path(path).stem}_{cfg.data.target_fps:g}fps.mp4"
        if not processed.exists():
            convert_video(path, processed, cfg.data.target_fps, tuple(cfg.data.resize))  # type: ignore[arg-type]
        else:
            logger.info("Reusing existing converted video %s", processed)
    video_id = db.upsert_video(
        path, info.fps, info.frame_count, info.width, info.height, info.duration_seconds,
        processed_path=processed,
    )
    logger.info(
        "Registered %s (id=%d, %.2f fps, %d frames, %.1fs)",
        Path(path).name, video_id, info.fps, info.frame_count, info.duration_seconds,
    )
    record = db.get_video(video_id)
    assert record is not None
    return record


IMPORTED_BY = "import"


def import_label_file(db: Database, video: VideoRecord, label_file: str | Path) -> int:
    """Load ranges from a CSV/JSON label file into the DB for ``video``.

    Re-importing is idempotent: ranges previously imported for this video are replaced,
    while ranges entered by hand in the labeler are left untouched.
    """
    ranges = load_label_file(label_file)
    for old in db.list_label_ranges(video.id):
        if old.labeled_by == IMPORTED_BY:
            db.delete_label_range(old.id)
    for r in ranges:
        db.add_label_range(video.id, r.start_sec, r.end_sec, r.label, labeled_by=IMPORTED_BY,
                           notes=f"from {Path(label_file).name}")
    logger.info("Imported %d ranges for %s from %s", len(ranges), video.name, label_file)
    return len(ranges)


def sidecar_label_file(video_path: str | Path) -> Optional[Path]:
    """Return ``<stem>.labels.csv`` / ``.labels.json`` next to the video if present."""
    p = Path(video_path)
    for suffix in (".labels.csv", ".labels.json"):
        candidate = p.with_name(p.stem + suffix)
        if candidate.exists():
            return candidate
    return None


def build_windows(db: Database, video: VideoRecord, cfg: Config) -> int:
    """(Re)compute clip windows for ``video`` and map its labeled ranges onto them."""
    ranges = [LabelRange(r.start_sec, r.end_sec, r.label) for r in db.list_label_ranges(video.id)]
    unknown = sorted({r.label for r in ranges} - set(cfg.labels.classes))
    if unknown:
        logger.warning("%s has range labels not in labels.classes: %s", video.name, unknown)
    windows = compute_windows(
        video.duration_seconds, cfg.data.window_seconds, cfg.data.stride_seconds,
        cfg.data.min_window_seconds,
    )
    labels = label_windows(windows, ranges, cfg.labels.default_class, cfg.labels.overlap_threshold)
    db.replace_clip_windows(video.id, [(w.index, w.start_sec, w.end_sec, lab) for w, lab in zip(windows, labels)])
    counts = {c: labels.count(c) for c in sorted(set(labels))}
    logger.info("%s: %d windows %s from %d labeled ranges", video.name, len(windows), counts, len(ranges))
    return len(windows)


def extract(
    db: Database,
    cfg: Config,
    inputs: Optional[Sequence[str | Path]] = None,
    convert: bool = False,
    labeled_only: bool = False,
    import_labels: bool = True,
) -> List[VideoRecord]:
    """Register videos (files or directories) and build their labeled clip windows.

    Args:
        db: database to write to.
        cfg: pipeline config.
        inputs: video files and/or directories. If empty, every video already in the
            database is re-windowed (useful after labeling).
        convert: write resampled copies at ``data.target_fps``.
        labeled_only: only build windows for videos that have at least one labeled range.
        import_labels: load ``<stem>.labels.csv``/``.json`` files found next to input videos.
    """
    videos: List[VideoRecord] = []
    if inputs:
        files: List[Path] = []
        for item in inputs:
            p = Path(item)
            files.extend(find_videos(p, cfg.data.supported_extensions) if p.is_dir() else [p])
        if not files:
            raise FileNotFoundError(f"No videos found in {list(map(str, inputs))}")
        videos = [register_video(db, f, cfg, convert=convert) for f in files]
        if import_labels:
            for f, video in zip(files, videos):
                label_file = sidecar_label_file(f)
                if label_file is not None:
                    import_label_file(db, video, label_file)
    else:
        videos = db.list_videos()
        if not videos:
            raise RuntimeError("No videos in the database; pass video files or a directory")
    built = 0
    for video in videos:
        if labeled_only and not db.list_label_ranges(video.id):
            logger.info("Skipping %s (no labeled ranges)", video.name)
            continue
        built += build_windows(db, video, cfg)
    logger.info("Extraction complete: %d videos, %d windows", len(videos), built)
    return videos
