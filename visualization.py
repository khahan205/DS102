"""
visualization.py
----------------
Tất cả hàm vẽ biểu đồ cho dự án SVM Pneumonia:
  - show_samples          : Hiển thị ảnh mẫu
  - plot_training_curve   : Loss & Validation F1 theo epoch
  - plot_confusion_matrix : Heatmap confusion matrix
  - plot_metrics_bar      : Bar chart Precision / Recall / F1 một mô hình
  - plot_comparison       : Grouped bar chart so sánh 2 mô hình
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


# ------------------------------------------------------------------ #
#  Hiển thị ảnh mẫu                                                  #
# ------------------------------------------------------------------ #

def show_samples(
    X:        np.ndarray,
    y:        np.ndarray,
    img_size: int = 128,
    n:        int = 6,
    title:    str = "Sample Images",
) -> None:
    """
    Hiển thị n ảnh mẫu từ tập dữ liệu.

    Args:
        X        : Ảnh đã flatten, shape (N, img_size*img_size).
        y        : Nhãn tương ứng (±1).
        img_size : Kích thước để reshape về 2-D.
        n        : Số ảnh cần hiển thị.
        title    : Tiêu đề figure.
    """
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes = axes.flatten()

    for i in range(n):
        img   = X[i].reshape(img_size, img_size)
        label = "PNEUMONIA" if y[i] == 1 else "NORMAL"
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(label, fontsize=11)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Training curve (Loss + Validation F1)                             #
# ------------------------------------------------------------------ #

def plot_training_curve(
    loss_history:    List[float],
    val_f1_history:  List[float] = None,
    title:           str         = "Training Curve",
) -> None:
    """
    Vẽ đường loss huấn luyện và (tuỳ chọn) F1 trên validation theo epoch.

    Args:
        loss_history   : Danh sách loss sau mỗi epoch.
        val_f1_history : Danh sách Validation F1 sau mỗi epoch (tuỳ chọn).
        title          : Tiêu đề biểu đồ.
    """
    epochs = range(1, len(loss_history) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, loss_history, "b-o", markersize=4, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    if val_f1_history:
        ax2 = ax1.twinx()
        ax2.plot(epochs, val_f1_history, "r-s", markersize=4, label="Validation F1")
        ax2.set_ylabel("F1-score", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.set_ylim(0, 1.05)

        # Gộp legend cả 2 trục
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    plt.title(title, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Confusion Matrix heatmap                                           #
# ------------------------------------------------------------------ #

def plot_confusion_matrix(
    cm:          np.ndarray,
    title:       str       = "Confusion Matrix",
    cmap:        str       = "Blues",
    class_names: List[str] = None,
) -> None:
    """
    Vẽ confusion matrix dạng heatmap.

    Args:
        cm          : Confusion matrix (2×2).
        title       : Tiêu đề biểu đồ.
        cmap        : Tên colormap của matplotlib/seaborn.
        class_names : Tên class (mặc định ['Normal', 'Pneumonia']).
    """
    if class_names is None:
        class_names = ["Normal", "Pneumonia"]

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
    )
    plt.title(title, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Bar chart một mô hình                                             #
# ------------------------------------------------------------------ #

def plot_metrics_bar(
    precision: float,
    recall:    float,
    f1:        float,
    title:     str  = "Model Metrics",
    color:     str  = "steelblue",
) -> None:
    """
    Vẽ bar chart Precision / Recall / F1 cho một mô hình.

    Args:
        precision : Precision score.
        recall    : Recall score.
        f1        : F1-score.
        title     : Tiêu đề biểu đồ.
        color     : Màu cột.
    """
    metrics = ["Precision", "Recall", "F1-score"]
    values  = [precision, recall, f1]

    plt.figure(figsize=(5, 4))
    bars = plt.bar(metrics, values, color=color, edgecolor="white", width=0.5)
    plt.ylim(0, 1.15)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    plt.title(title, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Grouped bar chart so sánh 2 mô hình                               #
# ------------------------------------------------------------------ #

def plot_comparison(
    metrics_custom:  List[float],
    metrics_sklearn: List[float],
    labels:          List[str] = None,
    title:           str       = "Custom SVM vs Sklearn SVM",
) -> None:
    """
    So sánh kết quả Custom SVM và Sklearn SVM bằng grouped bar chart.

    Args:
        metrics_custom  : [precision, recall, f1] của mô hình tự cài.
        metrics_sklearn : [precision, recall, f1] của sklearn.
        labels          : Tên chỉ số (mặc định ['Precision', 'Recall', 'F1-score']).
        title           : Tiêu đề biểu đồ.
    """
    if labels is None:
        labels = ["Precision", "Recall", "F1-score"]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, metrics_custom,  width, label="Custom SVM",  color="steelblue", edgecolor="white")
    bars2 = ax.bar(x + width / 2, metrics_sklearn, width, label="Sklearn SVM", color="tomato",    edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=12)

    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    plt.tight_layout()
    plt.show()
