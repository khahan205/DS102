"""
svm_sklearn.py
--------------
SVM dùng thư viện scikit-learn.
Bao gồm grid search tham số C trên tập validation và huấn luyện mô hình cuối.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from typing import List


def select_best_C(
    X_train:  np.ndarray,
    y_train:  np.ndarray,
    X_val:    np.ndarray,
    y_val:    np.ndarray,
    C_values: List[float] = None,
    kernel:   str         = "linear",
) -> float:
    """
    Tìm giá trị C tốt nhất dựa trên F1-score trên tập validation.

    Args:
        X_train  : Đặc trưng tập train.
        y_train  : Nhãn tập train (±1).
        X_val    : Đặc trưng tập validation.
        y_val    : Nhãn tập validation (±1).
        C_values : Danh sách C cần thử (mặc định [0.01, 0.1, 1, 10]).
        kernel   : Kernel SVM (mặc định 'linear').

    Returns:
        best_C (float): Giá trị C cho F1 cao nhất trên validation.
    """
    if C_values is None:
        C_values = [0.01, 0.1, 1, 10]

    best_C  = None
    best_f1 = -1.0

    print(f"\n{'C':>8}  {'Val F1':>8}")
    print("-" * 20)

    for C in C_values:
        clf = SVC(kernel=kernel, C=C)
        clf.fit(X_train, y_train)

        y_pred_val = clf.predict(X_val)
        val_f1 = f1_score(
            (y_val      == 1).astype(int),
            (y_pred_val == 1).astype(int),
            zero_division=0,
        )
        print(f"  {C:>6}  {val_f1:>8.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_C  = C

    print(f"\n  → Best C = {best_C}  (Val F1 = {best_f1:.4f})")
    return best_C


def train_sklearn_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C:       float = 1.0,
    kernel:  str   = "linear",
) -> SVC:
    """
    Huấn luyện mô hình SVC của scikit-learn.

    Args:
        X_train : Đặc trưng tập train.
        y_train : Nhãn tập train (±1).
        C       : Tham số regularisation.
        kernel  : Kernel SVM.

    Returns:
        clf (SVC): Mô hình đã được huấn luyện.
    """
    print(f"\nHuấn luyện Sklearn SVM  kernel='{kernel}'  C={C} ...")
    clf = SVC(kernel=kernel, C=C)
    clf.fit(X_train, y_train)
    print("  Hoàn tất.")
    return clf
