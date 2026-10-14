"""K-fold splitting that keeps a video's windows together when possible."""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

logger = logging.getLogger(__name__)

Fold = Tuple[np.ndarray, np.ndarray]


def make_folds(
    labels: Sequence[int], groups: Sequence[int], k: int, seed: int, group_by_video: bool = True
) -> List[Fold]:
    """Return ``k`` ``(train_idx, val_idx)`` folds.

    Overlapping windows from the same video are near-duplicates, so with
    ``group_by_video`` every video is kept in a single fold (``StratifiedGroupKFold``).
    If there are fewer videos than folds, it falls back to window-level stratified
    folds and logs a warning that validation scores will be optimistic.
    ``k`` is reduced if the rarest class has fewer than ``k`` examples.
    """
    y = np.asarray(labels)
    g = np.asarray(groups)
    counts = np.bincount(y)
    present = counts[counts > 0]
    if len(present) < 2:
        raise ValueError("Need at least two classes present to cross-validate")
    k_eff = min(k, int(present.min()))
    if k_eff < 2:
        raise ValueError(f"Rarest class has only {present.min()} example(s); need >= 2 for CV")
    if k_eff < k:
        logger.warning("Reducing k_folds from %d to %d (rarest class has %d examples)", k, k_eff, present.min())
    idx = np.arange(len(y))
    n_groups = len(np.unique(g))
    if group_by_video and n_groups >= k_eff:
        splitter = StratifiedGroupKFold(n_splits=k_eff, shuffle=True, random_state=seed)
        return list(splitter.split(idx, y, g))
    if group_by_video:
        logger.warning(
            "Only %d video(s) for %d folds: splitting by window instead. Overlapping windows "
            "from the same video can leak between train and validation; treat scores as optimistic.",
            n_groups, k_eff,
        )
    splitter2 = StratifiedKFold(n_splits=k_eff, shuffle=True, random_state=seed)
    return list(splitter2.split(idx, y))
