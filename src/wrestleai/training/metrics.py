"""Classification metrics computed from predicted probabilities."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: Sequence[int],
    probs: np.ndarray,
    classes: Sequence[str],
    positive_index: int = 1,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Accuracy, precision, recall, F1, ROC-AUC, PR-AUC and a confusion matrix.

    Binary (two classes): precision/recall/F1 are for ``positive_index``, predictions use
    ``threshold`` on its probability, and AUCs use that probability as the score.
    Multi-class: predictions are argmax, P/R/F1 are macro averages, and AUCs are
    one-vs-rest macro averages.

    AUCs are ``None`` when undefined (e.g. only one class present in ``y_true``).
    """
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    n_classes = len(classes)
    if p.ndim != 2 or p.shape[1] != n_classes or len(y) != len(p):
        raise ValueError(f"probs must be (N, {n_classes}) aligned with y_true; got {p.shape} vs {len(y)}")
    labels = list(range(n_classes))
    binary = n_classes == 2
    if binary:
        score = p[:, positive_index]
        y_pred = np.where(score >= threshold, positive_index, 1 - positive_index)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, y_pred, labels=[positive_index], average=None, zero_division=0
        )
        precision, recall, f1_score = float(prec[0]), float(rec[0]), float(f1[0])
    else:
        y_pred = p.argmax(axis=1)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y, y_pred, labels=labels, average="macro", zero_division=0
        )
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    present = np.unique(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if binary and len(present) == 2:
                y_pos = (y == positive_index).astype(int)
                roc_auc = float(roc_auc_score(y_pos, score))
                pr_auc = float(average_precision_score(y_pos, score))
            elif not binary and len(present) == n_classes:
                roc_auc = float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=labels))
                onehot = np.eye(n_classes)[y]
                pr_auc = float(average_precision_score(onehot, p, average="macro"))
        except ValueError as exc:
            logger.debug("AUC undefined: %s", exc)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)) if len(y) else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "classes": list(classes),
        "support": {c: int((y == i).sum()) for i, c in enumerate(classes)},
    }


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Human-readable multi-line summary for logs."""
    def fmt(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v:.4f}"

    lines = [
        f"n={metrics['n']}  support={metrics['support']}",
        f"accuracy={fmt(metrics['accuracy'])}  precision={fmt(metrics['precision'])}  "
        f"recall={fmt(metrics['recall'])}  f1={fmt(metrics['f1'])}",
        f"roc_auc={fmt(metrics['roc_auc'])}  pr_auc={fmt(metrics['pr_auc'])}",
        "confusion matrix (rows=true, cols=pred): " + ", ".join(metrics["classes"]),
    ]
    width = max(len(c) for c in metrics["classes"])
    for cls, row in zip(metrics["classes"], metrics["confusion_matrix"]):
        lines.append(f"  {cls:>{width}} " + " ".join(f"{v:6d}" for v in row))
    return "\n".join(lines)
