"""
metrics.py
----------
Tính và in các chỉ số đánh giá mô hình phân loại:
  Precision, Recall, F1-score, Confusion Matrix.
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from typing import Tuple


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Tính Precision, Recall, F1-score và Confusion Matrix.

    Nhãn phải là ±1 (PNEUMONIA = 1, NORMAL = -1).
    Hàm tự chuyển sang nhị phân 0/1 trước khi tính.

    Args:
        y_true : Nhãn thực tế.
        y_pred : Nhãn dự đoán.

    Returns:
        Tuple (precision, recall, f1, confusion_matrix).
    """
    y_true_bin = (y_true == 1).astype(int)
    y_pred_bin = (y_pred == 1).astype(int)

    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall    = recall_score(y_true_bin,    y_pred_bin, zero_division=0)
    f1        = f1_score(y_true_bin,        y_pred_bin, zero_division=0)
    cm        = confusion_matrix(y_true_bin, y_pred_bin)

    return precision, recall, f1, cm


def print_metrics(
    precision:  float,
    recall:     float,
    f1:         float,
    cm:         np.ndarray,
    model_name: str = "Model",
) -> None:
    """
    In kết quả đánh giá ra console theo dạng bảng đẹp.

    Args:
        precision  : Precision score.
        recall     : Recall score.
        f1         : F1-score.
        cm         : Confusion matrix (2×2).
        model_name : Tên mô hình hiển thị trong tiêu đề.
    """
    sep = "=" * 45
    print(f"\n{sep}")
    print(f"  Kết quả: {model_name}")
    print(sep)
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1-score   : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"               Pred Normal  Pred Pneumonia")
    print(f"  True Normal  {cm[0,0]:>11}  {cm[0,1]:>14}")
    print(f"  True Pneum.  {cm[1,0]:>11}  {cm[1,1]:>14}")
    print(sep)
