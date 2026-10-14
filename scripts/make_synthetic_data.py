#!/usr/bin/env python3
"""Generate tiny synthetic match videos with highlight label files.

Each video ``synthetic_match_XX.mp4`` gets a sidecar ``synthetic_match_XX.labels.csv``
(``start_sec,end_sec,label``) that ``wrestleai extract`` imports automatically.

Example:
    python scripts/make_synthetic_data.py --out data/synthetic/videos
    wrestleai extract data/synthetic/videos --config configs/test.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from wrestleai.data.synthetic import make_synthetic_dataset
from wrestleai.utils.logging import setup_logging

logger = logging.getLogger("make_synthetic_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic/videos"), help="output directory")
    parser.add_argument("--num-videos", type=int, default=4)
    parser.add_argument("--duration", type=float, default=24.0, help="seconds per video")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--size", type=int, nargs=2, default=[96, 96], metavar=("H", "W"))
    parser.add_argument("--highlights", type=int, default=2, help="highlight ranges per video")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    setup_logging("INFO")
    pairs = make_synthetic_dataset(args.out, args.num_videos, args.duration, args.fps, tuple(args.size),
                                   args.highlights, args.seed)
    logger.info("Generated %d synthetic videos in %s", len(pairs), args.out)


if __name__ == "__main__":
    main()
