"""
dt_numpy.py
===========
Cài đặt Decision Tree thuần NumPy cho bài toán phân loại đa lớp.
Sử dụng Gini impurity làm tiêu chí chia nhánh, thuật toán tìm điểm chia
được vector hoá hoàn toàn bằng cumulative sum để duyệt toàn bộ điểm chia
hợp lệ với chi phí thấp.
"""

import numpy as np


class Node:
    """
    Một node trong cây quyết định.

    Attributes
    ----------
    feature   : int | None  – chỉ số đặc trưng dùng để chia (None nếu là lá)
    threshold : float | None – ngưỡng chia (None nếu là lá)
    left      : Node | None – nhánh trái  (x[feature] <= threshold)
    right     : Node | None – nhánh phải (x[feature] >  threshold)
    value     : int | None  – nhãn dự đoán nếu là lá
    """

    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value

    def is_leaf(self) -> bool:
        """Trả về True nếu node hiện tại là lá."""
        return self.value is not None


class DecisionTree:
   

    def __init__(self, max_depth=10, min_samples_split=2,
                 max_features=None, random_state=None):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.rng               = np.random.RandomState(random_state)
        self.root              = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Huấn luyện cây trên tập dữ liệu (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
        y : np.ndarray, shape (N,)
        """
        self.n_features_ = X.shape[1]
        self._max_features = (
            self.n_features_ if self.max_features is None
            else int(self.max_features)
        )
        self.root = self._grow(X, y, depth=0)

    def _leaf_value(self, y: np.ndarray) -> int:
        """Trả về class chiếm đa số trong y (majority vote)."""
        vals, counts = np.unique(y, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Tìm điểm chia tốt nhất bằng kỹ thuật cumulative sum vector hoá.
        Duyệt toàn bộ điểm chia hợp lệ trên các đặc trưng được chọn.

        Returns
        -------
        best_feat : int | None    – chỉ số đặc trưng tốt nhất
        best_thr  : float | None  – ngưỡng chia tốt nhất
        """
        n        = len(y)
        cls_uniq = np.unique(y)
        n_cls    = len(cls_uniq)
        tot      = np.array([np.sum(y == c) for c in cls_uniq], dtype=np.float64)
        g_parent = 1.0 - float(np.dot(tot / n, tot / n))

        best_gain, best_feat, best_thr = -1.0, None, None
        feat_ids = self.rng.choice(self.n_features_, self._max_features, replace=False)

        for feat in feat_ids:
            col   = X[:, feat]
            order = np.argsort(col, kind='stable')
            s_col = col[order]
            s_y   = y[order]

            y_idx   = np.searchsorted(cls_uniq, s_y)
            one_hot = np.zeros((n, n_cls), dtype=np.float64)
            one_hot[np.arange(n), y_idx] = 1.0

            left_c  = np.cumsum(one_hot, axis=0)[:-1]
            right_c = tot - left_c

            n_l = np.arange(1, n, dtype=np.float64)
            n_r = n - n_l

            p_l  = left_c  / n_l[:, None]
            p_r  = right_c / n_r[:, None]
            g_l  = 1.0 - (p_l ** 2).sum(axis=1)
            g_r  = 1.0 - (p_r ** 2).sum(axis=1)
            gain = g_parent - (n_l / n) * g_l - (n_r / n) * g_r

            valid = s_col[:-1] != s_col[1:]
            if not valid.any():
                continue
            gain[~valid] = -np.inf

            i_best = int(np.argmax(gain))
            if gain[i_best] > best_gain:
                best_gain = float(gain[i_best])
                best_feat = int(feat)
                best_thr  = float((s_col[i_best] + s_col[i_best + 1]) / 2.0)

        return best_feat, best_thr

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Xây cây đệ quy đến khi đạt điều kiện dừng."""
        if (depth >= self.max_depth
                or len(y) < self.min_samples_split
                or len(np.unique(y)) == 1):
            return Node(value=self._leaf_value(y))

        feat, thr = self._best_split(X, y)
        if feat is None:
            return Node(value=self._leaf_value(y))

        mask  = X[:, feat] <= thr
        left  = self._grow(X[mask],  y[mask],  depth + 1)
        right = self._grow(X[~mask], y[~mask], depth + 1)
        return Node(feature=feat, threshold=thr, left=left, right=right)

    def _predict_one(self, x: np.ndarray, node: Node) -> int:
        """Đi xuống cây cho một mẫu để lấy nhãn."""
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán nhãn cho ma trận đặc trưng X.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        y_pred : np.ndarray, shape (N,)
        """
        return np.array([self._predict_one(x, self.root) for x in X])
