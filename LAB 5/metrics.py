"""
metrics.py
----------
Quantitative clustering evaluation, implemented with NumPy only (no sklearn).
Because cluster labels are arbitrary permutations of the true labels, supervised
metrics must be permutation-invariant. We provide:

- distortion          : the K-means objective J = sum_n ||x_n - mu_{c_n}||^2
- contingency_matrix  : co-occurrence counts between true and predicted labels
- best_map_accuracy   : accuracy under the best label permutation (brute force)
- purity              : fraction of points in the majority true-class of their cluster
- adjusted_rand_index : ARI, chance-corrected agreement in [-1, 1] (1 = perfect)
"""

import numpy as np


def _permutations(seq):
    """All permutations of a short sequence, written from scratch (no imports).

    Used only by best_map_accuracy for small K, so the whole evaluation stays
    dependency-free (NumPy only)."""
    seq = list(seq)
    if len(seq) <= 1:
        yield tuple(seq)
        return
    for i in range(len(seq)):
        rest = seq[:i] + seq[i + 1:]
        for tail in _permutations(rest):
            yield (seq[i],) + tail


def distortion(X, labels, centers):
    """K-means objective J = sum over points of squared distance to its centroid."""
    X = np.asarray(X, float)
    diff = X - centers[labels]
    return float((diff ** 2).sum())


def contingency_matrix(y_true, y_pred):
    """Counts C[i, j] = #points with true label i and predicted label j."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = np.unique(y_true)
    cols = np.unique(y_pred)
    r_idx = {v: i for i, v in enumerate(rows)}
    c_idx = {v: i for i, v in enumerate(cols)}
    C = np.zeros((len(rows), len(cols)), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[r_idx[t], c_idx[p]] += 1
    return C


def best_map_accuracy(y_true, y_pred):
    """Accuracy under the optimal one-to-one mapping of predicted -> true labels.

    Cluster ids are arbitrary, so we try every permutation of the predicted labels
    onto the true labels and keep the best. Exact for small K (K! permutations).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_lbls = np.unique(y_true)
    pred_lbls = np.unique(y_pred)
    K = max(len(true_lbls), len(pred_lbls))
    best = 0.0
    for perm in _permutations(range(K)):
        mapping = {p: perm[i] for i, p in enumerate(pred_lbls)}
        mapped = np.array([mapping[p] for p in y_pred])
        acc = (mapped == y_true).mean()
        best = max(best, acc)
    return float(best)


def purity(y_true, y_pred):
    """Sum over clusters of the size of the majority true class, divided by N."""
    C = contingency_matrix(y_true, y_pred)
    return float(C.max(axis=0).sum() / C.sum())


def adjusted_rand_index(y_true, y_pred):
    """Adjusted Rand Index: 1 = identical clustering, ~0 = random, can be negative."""
    C = contingency_matrix(y_true, y_pred).astype(float)
    n = C.sum()

    def comb2(x):
        return x * (x - 1.0) / 2.0

    sum_ij = comb2(C).sum()
    sum_a = comb2(C.sum(axis=1)).sum()
    sum_b = comb2(C.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb2(n)
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom == 0:                       # degenerate (all in one cluster)
        return 1.0
    return float((sum_ij - expected) / denom)


def clustering_report(y_true, y_pred, X=None, centers=None):
    """Convenience: return a dict of all the metrics above."""
    rep = {
        "accuracy": best_map_accuracy(y_true, y_pred),
        "ARI": adjusted_rand_index(y_true, y_pred),
        "purity": purity(y_true, y_pred),
    }
    if X is not None and centers is not None:
        rep["distortion"] = distortion(X, y_pred, centers)
    return rep
