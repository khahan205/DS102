"""
kmeans.py
--------
K-means clustering implemented with NumPy only, trained with the EM (Lloyd)
algorithm:

    Objective (distortion)
        J = sum_n sum_k r_nk * ||x_n - mu_k||^2

    E-step  : assign each point to its nearest centroid (r_nk = 1 for that k)
    M-step  : recompute each centroid as the mean of the points assigned to it

The two steps alternate until the centroids stop moving (or max_iter is hit).
"""

import numpy as np


class KMeans:
    """K-means clustering trained with the EM (Lloyd) algorithm."""

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-8, random_state=None, init=None):
        self.K = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init                 # None -> random data points; or an (K, D) array

    # ------------------------------------------------------------------ helpers
    def _init_centroids(self, X):
        if self.init is not None:
            return np.asarray(self.init, float).copy()
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), self.K, replace=False)      # K distinct data points
        return X[idx].copy()

    @staticmethod
    def _sq_dist(X, C):
        """Squared Euclidean distances between points and centroids -> (N, K)."""
        return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)

    # ------------------------------------------------------------------ training
    def fit(self, X):
        X = np.asarray(X, float)
        C = self._init_centroids(X)
        self.history_ = []                                   # distortion J per iteration
        for it in range(self.max_iter):
            # ---- E-step: assign every point to the nearest centroid ----
            D = self._sq_dist(X, C)
            labels = D.argmin(axis=1)
            J = D[np.arange(len(X)), labels].sum()
            self.history_.append(J)

            # ---- M-step: recompute each centroid as the mean of its members ----
            newC = C.copy()
            for k in range(self.K):
                members = X[labels == k]
                if len(members):
                    newC[k] = members.mean(axis=0)
                else:                                        # empty cluster -> re-seed
                    newC[k] = X[D.min(axis=1).argmax()]      # to the farthest point

            shift = np.sqrt(((newC - C) ** 2).sum())
            C = newC
            if shift < self.tol:                             # converged
                break

        self.cluster_centers_ = C
        D = self._sq_dist(X, C)
        self.labels_ = D.argmin(axis=1)
        self.inertia_ = D[np.arange(len(X)), self.labels_].sum()
        self.n_iter_ = it + 1
        return self

    def predict(self, X):
        return self._sq_dist(np.asarray(X, float), self.cluster_centers_).argmin(axis=1)
