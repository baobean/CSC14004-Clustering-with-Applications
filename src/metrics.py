"""Clustering evaluation metrics.

The rubric (§3.2.4) requires at least three metrics. For labeled data
we expose Accuracy (with Hungarian label matching), NMI, and ARI; we
additionally expose misclustering error, which is the primary metric
used by the paper (Tables 2-3).

All functions accept 1-D array-likes of integer cluster assignments.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def accuracy_hungarian(y_true, y_pred) -> float:
    """Clustering accuracy after optimal label permutation (Hungarian).

    Parameters
    ----------
    y_true, y_pred : array-like of int
        Ground-truth and predicted cluster indices (need not share
        the same label space; they are aligned via Hungarian matching
        on the confusion matrix).

    Returns
    -------
    float
        Fraction of points whose predicted cluster, after the optimal
        permutation, matches the ground-truth cluster.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape
    n_classes = max(y_true.max(), y_pred.max()) + 1
    w = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        w[t, p] += 1
    # Hungarian maximizes when we negate.
    row_ind, col_ind = linear_sum_assignment(-w)
    return w[row_ind, col_ind].sum() / y_true.size


def nmi(y_true, y_pred) -> float:
    """Normalized mutual information (sklearn)."""
    return float(normalized_mutual_info_score(y_true, y_pred))


def ari(y_true, y_pred) -> float:
    """Adjusted Rand index (sklearn)."""
    return float(adjusted_rand_score(y_true, y_pred))


def misclustering_error(y_true, y_pred) -> float:
    """1 - accuracy_hungarian. Matches the paper's Tables 2-3 metric."""
    return 1.0 - accuracy_hungarian(y_true, y_pred)
