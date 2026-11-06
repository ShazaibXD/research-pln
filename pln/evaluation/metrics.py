"""Evaluation metrics (NumPy). Inputs are validated; ties are handled explicitly.

Note (derivation.md section 11.1): precision, AUPRC and calibration are only interpretable on labels
that are not positive-unlabeled, i.e. an audit sample or fully known (synthetic) labels.
"""

from __future__ import annotations

import numpy as np

from pln.errors import PLNInputError


def _check(y: np.ndarray, s: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise PLNInputError("y_true and scores must be 1-D arrays of equal length")
    if not np.isfinite(s).all():
        raise PLNInputError("scores must be finite")
    if not np.isin(y, (0, 1)).all():
        raise PLNInputError("y_true must be binary {0,1}")
    return y.astype(np.int64), s


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area under the PR curve by step interpolation); nan if there are no positives."""
    y, s = _check(y_true, scores)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    # only evaluate at the end of each tied-score block
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via rank statistics with average ranks for ties; nan if only one class is present."""
    y, s = _check(y_true, scores)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    y, p = _check(y_true, probs)
    if (p < 0).any() or (p > 1).any():
        raise PLNInputError("probabilities must lie in [0, 1]")
    return float(np.mean((p - y) ** 2))


def log_loss(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-12) -> float:
    y, p = _check(y_true, probs)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE: sum_b (n_b/n) |mean(y) - mean(p)|."""
    y, p = _check(y_true, probs)
    if (p < 0).any() or (p > 1).any():
        raise PLNInputError("probabilities must lie in [0, 1]")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            ece += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(ece)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    y, s = _check(y_true, scores)
    if k < 1:
        raise PLNInputError("k must be >= 1")
    k = min(k, len(y))
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].mean())
