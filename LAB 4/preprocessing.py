"""
preprocessing.py
================
Hàm chia tập dữ liệu thành train / test theo stratified sampling, thuần NumPy.
"""

import numpy as np


def numpy_train_test_split(
    X:            np.ndarray,
    y:            np.ndarray,
    test_size:    float = 0.2,
    random_state: int   = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chia tập dữ liệu thành train / test, giữ nguyên tỷ lệ các class
    (stratified sampling) – cài đặt thuần NumPy, không dùng sklearn.

    Parameters
    ----------
    X            : np.ndarray, shape (N, D) – ma trận đặc trưng
    y            : np.ndarray, shape (N,)   – vector nhãn
    test_size    : float                    – tỷ lệ tập test (mặc định 0.2)
    random_state : int                      – seed cho RandomState

    Returns
    -------
    X_train : np.ndarray, shape (N_train, D)
    X_test  : np.ndarray, shape (N_test,  D)
    y_train : np.ndarray, shape (N_train,)
    y_test  : np.ndarray, shape (N_test,)
    """
    rng     = np.random.RandomState(random_state)
    classes = np.unique(y)

    train_idx, test_idx = [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test_size))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])

    tr = np.concatenate(train_idx)
    te = np.concatenate(test_idx)
    rng.shuffle(tr)
    rng.shuffle(te)

    return X[tr], X[te], y[tr], y[te]
