"""
metrics.py
==========
Các hàm đánh giá thuần NumPy: F1 Score, Confusion Matrix, Classification Report.
"""

import numpy as np


def numpy_f1_score(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    average: str | None = 'weighted',
) -> float | np.ndarray:
    """
    Tính F1 Score thuần NumPy, hỗ trợ 3 chế độ aggregation.

    Parameters
    ----------
    y_true  : np.ndarray, shape (N,) – nhãn thực tế
    y_pred  : np.ndarray, shape (N,) – nhãn dự đoán
    average : {'weighted', 'macro', None}
        - 'weighted' : trung bình có trọng số theo support
        - 'macro'    : trung bình không trọng số
        - None       : trả về F1 từng class

    Returns
    -------
    float | np.ndarray
        - float nếu average ∈ {'weighted', 'macro'}
        - np.ndarray shape (n_classes,) nếu average is None
    """
    cls_unique = np.unique(y_true)
    f1s        = np.zeros(len(cls_unique))
    supports   = np.zeros(len(cls_unique), dtype=int)

    for i, c in enumerate(cls_unique):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s[i]      = (2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
        supports[i] = int(np.sum(y_true == c))

    if average == 'weighted':
        return float(np.dot(f1s, supports) / supports.sum())
    if average == 'macro':
        return float(np.mean(f1s))
    return f1s


def numpy_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
) -> np.ndarray:
    """
    Tính confusion matrix thuần NumPy.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,) – nhãn thực tế
    y_pred : np.ndarray, shape (N,) – nhãn dự đoán
    labels : list[int]              – danh sách class cần tính, đúng thứ tự

    Returns
    -------
    cm : np.ndarray, shape (n_labels, n_labels), dtype int
        cm[i, j] = số mẫu thuộc class labels[i] được dự đoán là labels[j]
    """
    idx_map = {lbl: i for i, lbl in enumerate(labels)}
    cm      = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx_map and p in idx_map:
            cm[idx_map[t], idx_map[p]] += 1
    return cm


def numpy_classification_report(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    classes: list[int],
) -> None:
    """
    In bảng báo cáo Precision / Recall / F1-Score / Support cho từng class,
    cùng với weighted average và macro average ở cuối.

    Parameters
    ----------
    y_true  : np.ndarray, shape (N,)
    y_pred  : np.ndarray, shape (N,)
    classes : list[int] – danh sách class cần báo cáo

    Returns
    -------
    None – chỉ in kết quả ra stdout
    """
    header = f"{'Class':>12}  {'Precision':>10}  {'Recall':>8}  {'F1-Score':>9}  {'Support':>8}"
    sep    = '-' * len(header)
    print(header)
    print(sep)

    f1s, sups = [], []
    for cls in classes:
        tp  = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp  = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn  = int(np.sum((y_true == cls) & (y_pred != cls)))
        sup = int(np.sum(y_true == cls))
        pr  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1  = (2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
        print(f"{'Q'+str(cls):>12}  {pr:>10.4f}  {rc:>8.4f}  {f1:>9.4f}  {sup:>8}")
        f1s.append(f1)
        sups.append(sup)

    f1s  = np.array(f1s)
    sups = np.array(sups)
    print(sep)
    print(f"{'weighted avg':>12}  {'':>10}  {'':>8}  {np.dot(f1s, sups) / sups.sum():>9.4f}  {sups.sum():>8}")
    print(f"{'macro avg':>12}  {'':>10}  {'':>8}  {np.mean(f1s):>9.4f}  {sups.sum():>8}")
