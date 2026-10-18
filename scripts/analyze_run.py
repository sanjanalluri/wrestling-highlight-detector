#!/usr/bin/env python3
"""Detailed out-of-fold analysis of one training run, or a comparison of two.

Every window is scored by the fold model that held it out, so nothing is scored by a
model that trained on its match. On top of what ``wrestleai evaluate`` reports, this shows:

* window-level accuracy / precision / recall / F1 at several thresholds, next to the
  accuracy of always predicting the default class (the baseline accuracy has to beat)
* event-level results: how many labeled positive ranges overlap a detected range, and
  how many detected ranges overlap no label (false alarms, also per hour of footage)
* with two runs: a bootstrap over matches of the ROC-AUC / PR-AUC difference, and the
  PR-AUC of each match under both runs

Examples:
    python scripts/analyze_run.py                       # latest run
    python scripts/analyze_run.py real16-b3-motion
    python scripts/analyze_run.py real16-b3-lr1e-4 real16-b3-motion --json outputs/compare.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from wrestleai.config import load_config
from wrestleai.data.dataset import samples_from_db
from wrestleai.data.windowing import Window
from wrestleai.db.database import Database
from wrestleai.inference.predict import merge_positive_windows
from wrestleai.training.checkpoint import load_checkpoint, resolve_run_dir
from wrestleai.training.trainer import score_samples
from wrestleai.utils.logging import setup_logging
from wrestleai.utils.runtime import resolve_device

logger = logging.getLogger("analyze_run")

# window_id -> (video_id, start_sec, end_sec, is_positive, p_positive)
Rows = Dict[int, Tuple[int, float, float, int, float]]


def out_of_fold(run_dir: Path, db: Database, device_name: str) -> Tuple[Rows, str]:
    """Score every validation window of every fold with that fold's model."""
    device = resolve_device(device_name)
    rows: Rows = {}
    positive = ""
    folds = sorted(run_dir.glob("fold_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not folds:
        raise FileNotFoundError(f"No fold checkpoints in {run_dir}")
    for path in folds:
        model, cfg, ckpt = load_checkpoint(path, device)
        positive = cfg.labels.positive_class
        pos_idx = cfg.labels.classes.index(positive)
        samples = samples_from_db(db, cfg, window_ids=ckpt["val_window_ids"])
        probs = score_samples(model, samples, cfg, device)[:, pos_idx]
        for s, p in zip(samples, probs):
            rows[int(s.window_id)] = (int(s.video_id), s.start_sec, s.end_sec, int(s.label == pos_idx), float(p))
    return rows, positive


def window_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"accuracy": float((pred == (y == 1)).mean()), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9), "tp": tp, "fp": fp, "fn": fn}


def event_metrics(rows: Rows, db: Database, positive: str, thr: float, max_gap: float) -> Dict[str, float]:
    """Labeled positive ranges hit by a detected range, and detected ranges matching no label."""
    by_video: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
    for video_id, start, end, _, p in rows.values():
        by_video[video_id].append((start, end, p))
    hit = total = false_alarms = 0
    hours = 0.0
    for video_id, wins in by_video.items():
        video = db.get_video(video_id)
        hours += (video.duration_seconds if video else 0.0) / 3600
        labeled = [r for r in db.list_label_ranges(video_id) if r.label == positive]
        wins.sort()
        detected = merge_positive_windows([Window(i, s, e) for i, (s, e, _) in enumerate(wins)],
                                          [p for _, _, p in wins], thr, max_gap)
        overlaps = lambda a, b: a.start_sec < b.end_sec and a.end_sec > b.start_sec  # noqa: E731
        total += len(labeled)
        hit += sum(any(overlaps(d, r) for d in detected) for r in labeled)
        false_alarms += sum(not any(overlaps(d, r) for r in labeled) for d in detected)
    return {"found": hit, "labeled": total, "found_rate": hit / max(total, 1),
            "false_alarms": false_alarms, "false_alarms_per_hour": false_alarms / max(hours, 1e-9),
            "hours": hours}


def summarize(name: str, rows: Rows, db: Database, positive: str, thresholds: Sequence[float],
              max_gap: float) -> Dict[str, Any]:
    y = np.array([r[3] for r in rows.values()])
    p = np.array([r[4] for r in rows.values()])
    both = len(np.unique(y)) == 2
    report: Dict[str, Any] = {
        "run": name, "windows": int(len(y)), "positive_windows": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "chance_pr_auc": float(y.mean()), "always_negative_accuracy": float(1 - y.mean()),
        "thresholds": [],
    }
    for thr in thresholds:
        report["thresholds"].append({"threshold": thr, **window_metrics(y, p, thr),
                                     **event_metrics(rows, db, positive, thr, max_gap)})
    return report


def print_report(r: Dict[str, Any], positive: str) -> None:
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    print(f"\n===== {r['run']}: {r['windows']} windows, {r['positive_windows']} {positive} "
          f"({r['chance_pr_auc']:.1%}) over {r['thresholds'][0]['hours']:.2f} h of footage =====")
    print(f"ROC-AUC {fmt(r['roc_auc'])}   PR-AUC {fmt(r['pr_auc'])}   (chance PR-AUC {r['chance_pr_auc']:.3f}, "
          f"always-negative accuracy {r['always_negative_accuracy']:.3f})")
    print("thr    acc    prec   rec    f1      TP    FP  | ranges found     false alarms (per hour)")
    for t in r["thresholds"]:
        print(f"{t['threshold']:.2f}  {t['accuracy']:.3f}  {t['precision']:.3f}  {t['recall']:.3f}  {t['f1']:.3f}  "
              f"{t['tp']:4d}  {t['fp']:4d}  |  {t['found']:3d}/{t['labeled']} ({t['found_rate']:.0%})"
              f"     {t['false_alarms']:4d} ({t['false_alarms_per_hour']:.0f}/h)")


def compare(a: Tuple[str, Rows], b: Tuple[str, Rows], db: Database, n_boot: int, seed: int) -> Dict[str, Any]:
    """Bootstrap over matches of run B minus run A, plus per-match PR-AUC."""
    (name_a, rows_a), (name_b, rows_b) = a, b
    common = sorted(set(rows_a) & set(rows_b))
    by_video: Dict[int, List[int]] = defaultdict(list)
    for w in common:
        by_video[rows_a[w][0]].append(w)
    videos = sorted(by_video)
    rng = np.random.default_rng(seed)
    diffs: Dict[str, List[float]] = {"roc_auc": [], "pr_auc": []}
    for _ in range(n_boot):
        windows = [w for v in rng.choice(videos, size=len(videos), replace=True) for w in by_video[v]]
        y = np.array([rows_a[w][3] for w in windows])
        if y.min() == y.max():
            continue
        pa = np.array([rows_a[w][4] for w in windows])
        pb = np.array([rows_b[w][4] for w in windows])
        diffs["roc_auc"].append(roc_auc_score(y, pb) - roc_auc_score(y, pa))
        diffs["pr_auc"].append(average_precision_score(y, pb) - average_precision_score(y, pa))
    result: Dict[str, Any] = {"a": name_a, "b": name_b, "bootstrap": {}, "per_match_pr_auc": {}}
    print(f"\n===== {name_b} minus {name_a} (bootstrap over {len(videos)} matches, {n_boot} resamples) =====")
    for key, values in diffs.items():
        d = np.array(values)
        stats = {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                 "p_improvement": float((d > 0).mean())}
        result["bootstrap"][key] = stats
        print(f"{key:8s} {stats['mean']:+.3f}   95% CI [{stats['ci95'][0]:+.3f}, {stats['ci95'][1]:+.3f}]   "
              f"P(B better) = {stats['p_improvement']:.0%}")
    print("\nPer-match PR-AUC (matches with at least one positive window):")
    for v in videos:
        y = np.array([rows_a[w][3] for w in by_video[v]])
        if y.sum() == 0:
            continue
        ap_a = float(average_precision_score(y, [rows_a[w][4] for w in by_video[v]]))
        ap_b = float(average_precision_score(y, [rows_b[w][4] for w in by_video[v]]))
        video = db.get_video(v)
        name = video.name if video else str(v)
        result["per_match_pr_auc"][name] = {"a": ap_a, "b": ap_b}
        print(f"  {name:28s} {ap_a:.3f} -> {ap_b:.3f}  ({ap_b - ap_a:+.3f})")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="one or two run names/directories (default: latest run)")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--bootstrap", type=int, default=2000, help="resamples for the two-run comparison")
    parser.add_argument("--json", type=Path, default=None, help="also write the results to this JSON file")
    args = parser.parse_args(argv)
    setup_logging("WARNING")
    cfg = load_config(args.config if args.config.exists() else None, args.overrides)
    if len(args.runs) > 2:
        parser.error("pass at most two runs")
    db = Database(cfg.paths.db_path)
    run_dirs = [resolve_run_dir(cfg.paths.checkpoint_dir, r) for r in (args.runs or [None])]
    results: Dict[str, Any] = {"runs": []}
    scored: List[Tuple[str, Rows]] = []
    for run_dir in run_dirs:
        rows, positive = out_of_fold(run_dir, db, cfg.training.device)
        report = summarize(run_dir.name, rows, db, positive, args.thresholds, cfg.inference.merge_gap_seconds)
        print_report(report, positive)
        results["runs"].append(report)
        scored.append((run_dir.name, rows))
    if len(scored) == 2:
        results["comparison"] = compare(scored[0], scored[1], db, args.bootstrap, cfg.training.seed)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
