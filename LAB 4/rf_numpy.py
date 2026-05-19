"""
rf_numpy.py
===========
Cài đặt Random Forest thuần NumPy bằng cách tổ hợp nhiều DecisionTree:
- Mỗi cây huấn luyện trên một bootstrap sample (lấy mẫu có hoàn lại).
- Mỗi cây chỉ xét một tập con đặc trưng ngẫu nhiên tại mỗi split.
- Dự đoán cuối cùng = majority vote của tất cả cây.
"""

import numpy as np
from dt_numpy import DecisionTree


class RandomForest:
    """
    Random Forest Classifier thuần NumPy.

    Parameters
    ----------
    n_estimators      : int – số cây trong rừng
    max_depth         : int – độ sâu tối đa của mỗi cây
    min_samples_split : int – số mẫu tối thiểu để chia một node
    max_features      : {'sqrt', 'log2', int, None}
        Số đặc trưng được xét tại mỗi split:
            - 'sqrt' : sqrt(n_features)
            - 'log2' : log2(n_features)
            - int    : số cụ thể
            - None   : dùng tất cả
    random_state      : int | None – seed cho việc bootstrap + chọn đặc trưng
    """

    def __init__(self, n_estimators=50, max_depth=8, min_samples_split=5,
                 max_features='sqrt', random_state=None):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.rng               = np.random.RandomState(random_state)
        self.trees_            = []

    def _resolve_max_features(self, n_features: int) -> int:
        """Quy đổi max_features về số nguyên cụ thể."""
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        if self.max_features is None:
            return n_features
        return int(self.max_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Huấn luyện rừng bằng cách build n_estimators cây con,
        mỗi cây trên một bootstrap sample khác nhau.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
        y : np.ndarray, shape (N,)
        """
        n_samples, n_features = X.shape
        max_feat    = self._resolve_max_features(n_features)
        self.trees_ = []

        for i in range(self.n_estimators):
            seed    = self.rng.randint(0, 2 ** 31 - 1)
            indices = self.rng.choice(n_samples, n_samples, replace=True)
            tree    = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feat,
                random_state=seed,
            )
            tree.fit(X[indices], y[indices])
            self.trees_.append(tree)
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{self.n_estimators}] trees built")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán bằng majority vote từ tất cả các cây.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        y_pred : np.ndarray, shape (N,)
        """
        all_preds = np.stack([t.predict(X) for t in self.trees_], axis=0)
        result    = np.empty(X.shape[0], dtype=all_preds.dtype)
        for i in range(X.shape[0]):
            vals, counts = np.unique(all_preds[:, i], return_counts=True)
            result[i]    = vals[np.argmax(counts)]
        return result
