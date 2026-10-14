"""Command-line entry point: ``wrestleai {label,extract,train,evaluate,infer}``."""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from wrestleai.config import Config, load_config
from wrestleai.utils.logging import setup_logging

logger = logging.getLogger("wrestleai.cli")

DEFAULT_CONFIG = Path("configs") / "default.yaml"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None,
                        help=f"YAML config (default: {DEFAULT_CONFIG} if it exists, else built-in defaults)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set training.epochs=10 (repeatable)")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING... (overrides config)")
    parser.add_argument("--log-file", type=Path, default=None, help="also write logs to this file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wrestleai",
        description="Groot pipeline: DeepLabV3+ masking -> EfficientNet features -> Bi-LSTM -> "
                    "highlight classifier for wrestling footage.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("label", help="mark labeled time ranges for a video (interactive terminal tool)",
                       description="Label highlight/technique time ranges for one video. Runs interactively "
                                   "unless --add/--import/--list is given. Clip windows for the video are "
                                   "rebuilt afterwards.")
    _add_common(p)
    p.add_argument("video", help="video file (registered automatically) or name of a registered video")
    p.add_argument("--add", nargs="+", action="append", default=[], metavar="ARG",
                   help="non-interactive: START END [LABEL], e.g. --add 1:30 1:42 highlight (repeatable)")
    p.add_argument("--import", dest="import_file", type=Path, help="import ranges from a CSV/JSON label file")
    p.add_argument("--list", action="store_true", help="print stored ranges and exit")
    p.add_argument("--labeler", default=None, help="name recorded with each range (default: OS user)")

    p = sub.add_parser("extract", help="register videos, optionally resample them, and build labeled clip windows",
                       description="Probe videos with OpenCV, store them in SQLite, import sidecar "
                                   "<name>.labels.csv files, and slice each video into labeled windows. "
                                   "With no inputs, re-windows every video already in the database.")
    _add_common(p)
    p.add_argument("inputs", nargs="*", help="video files and/or directories (default: none = all DB videos)")
    p.add_argument("--video-dir", action="store_true", help="use paths.video_dir from the config as input")
    p.add_argument("--convert", action="store_true",
                   help="write resampled copies at data.target_fps / data.resize for faster training")
    p.add_argument("--labeled-only", action="store_true", help="only build windows for videos with labels")
    p.add_argument("--no-import-labels", action="store_true", help="ignore sidecar <name>.labels.csv files")

    p = sub.add_parser("train", help="k-fold cross-validated training",
                       description="Train with k-fold cross-validation over all labeled windows. Writes "
                                   "fold checkpoints, best.pt and cv_summary.json to paths.checkpoint_dir/<run>.")
    _add_common(p)
    p.add_argument("--run-name", default=None, help="run directory name (default: run-<timestamp>)")
    p.add_argument("--video-ids", type=int, nargs="+", default=None, help="restrict training to these video ids")

    p = sub.add_parser("evaluate", help="accuracy/precision/recall/F1/ROC-AUC/PR-AUC + confusion matrix",
                       description="Default: out-of-fold evaluation of a training run (each window scored by "
                                   "the fold model that held it out). With --checkpoint and --video-ids: score "
                                   "one checkpoint on specific (held-out) videos.")
    _add_common(p)
    p.add_argument("--run", default=None, help="run name or directory (default: latest run)")
    p.add_argument("--checkpoint", type=Path, default=None, help="evaluate a single checkpoint instead")
    p.add_argument("--video-ids", type=int, nargs="+", default=None, help="videos to score with --checkpoint")

    p = sub.add_parser("infer", help="per-window highlight probabilities and detected time ranges for a video",
                       description="Score every window of a video and merge adjacent positive windows "
                                   "into highlight time ranges.")
    _add_common(p)
    p.add_argument("video", type=Path, help="video file")
    p.add_argument("--checkpoint", type=Path, default=None, help="model checkpoint (default: best.pt of --run)")
    p.add_argument("--run", default=None, help="run name or directory (default: latest run)")
    p.add_argument("--threshold", type=float, default=None, help="override inference.threshold")
    p.add_argument("--out-dir", type=Path, default=None, help="output directory (default: paths.output_dir)")
    return parser


def _load_cfg(args: argparse.Namespace) -> Config:
    path = args.config
    if path is None and DEFAULT_CONFIG.exists():
        path = DEFAULT_CONFIG
    cfg = load_config(path, args.overrides)
    setup_logging(args.log_level or cfg.log_level, args.log_file)
    logger.debug("Config: %s", json.dumps(cfg.to_dict()))
    return cfg


# ----------------------------------------------------------------- commands
def cmd_label(args: argparse.Namespace, cfg: Config) -> int:
    from wrestleai.data.extract import build_windows, import_label_file, register_video
    from wrestleai.db.database import Database
    from wrestleai.labeling.range_labeler import RangeLabeler
    from wrestleai.utils.timefmt import parse_time

    db = Database(cfg.paths.db_path)
    video = db.find_video(args.video)
    if video is None:
        if not Path(args.video).is_file():
            logger.error("%s is neither a registered video nor an existing file", args.video)
            return 2
        video = register_video(db, args.video, cfg)
    labeler = RangeLabeler(db, video, cfg.labels.classes, cfg.labels.positive_class,
                           labeled_by=args.labeler or getpass.getuser())
    if args.list:
        labeler.show_ranges()
        return 0
    if args.import_file:
        import_label_file(db, video, args.import_file)
    for spec in args.add:
        if len(spec) not in (2, 3):
            logger.error("--add expects START END [LABEL], got %s", spec)
            return 2
        try:
            start, end = parse_time(spec[0]), parse_time(spec[1])
        except ValueError as exc:
            logger.error("%s", exc)
            return 2
        label = spec[2] if len(spec) == 3 else cfg.labels.positive_class
        if labeler.add(start, end, label, confirm_overlap=False) is None:
            return 2
    if not args.add and not args.import_file:
        labeler.run()
    build_windows(db, video, cfg)
    return 0


def cmd_extract(args: argparse.Namespace, cfg: Config) -> int:
    from wrestleai.data.extract import extract
    from wrestleai.db.database import Database

    inputs: List[str] = list(args.inputs)
    if args.video_dir:
        inputs.append(cfg.paths.video_dir)
    db = Database(cfg.paths.db_path)
    extract(db, cfg, inputs, convert=args.convert, labeled_only=args.labeled_only,
            import_labels=not args.no_import_labels)
    return 0


def cmd_train(args: argparse.Namespace, cfg: Config) -> int:
    from wrestleai.db.database import Database
    from wrestleai.training.trainer import run_training

    run_training(cfg, Database(cfg.paths.db_path), args.run_name, args.video_ids)
    return 0


def cmd_evaluate(args: argparse.Namespace, cfg: Config) -> int:
    from wrestleai.db.database import Database
    from wrestleai.training.checkpoint import resolve_run_dir
    from wrestleai.training.evaluate import evaluate_checkpoint, evaluate_run

    db = Database(cfg.paths.db_path)
    if args.checkpoint is not None:
        if not args.video_ids:
            logger.error("--checkpoint requires --video-ids (the videos to evaluate on)")
            return 2
        evaluate_checkpoint(args.checkpoint, db, args.video_ids, cfg.paths.output_dir, cfg.training.device)
    else:
        evaluate_run(resolve_run_dir(cfg.paths.checkpoint_dir, args.run), db, cfg.training.device)
    return 0


def cmd_infer(args: argparse.Namespace, cfg: Config) -> int:
    from wrestleai.inference.predict import infer_video
    from wrestleai.training.checkpoint import resolve_run_dir

    checkpoint = args.checkpoint or resolve_run_dir(cfg.paths.checkpoint_dir, args.run) / "best.pt"
    infer_video(args.video, checkpoint, args.out_dir or cfg.paths.output_dir, args.threshold, cfg.training.device)
    return 0


COMMANDS = {"label": cmd_label, "extract": cmd_extract, "train": cmd_train,
            "evaluate": cmd_evaluate, "infer": cmd_infer}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = _load_cfg(args)
    except (OSError, ValueError) as exc:
        setup_logging("INFO")
        logger.error("Invalid configuration: %s", exc)
        return 2
    try:
        return COMMANDS[args.command](args, cfg)
    except (FileNotFoundError, RuntimeError, IOError, KeyError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
