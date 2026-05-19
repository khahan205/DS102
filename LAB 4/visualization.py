"""
visualization.py
================
Các hàm vẽ biểu đồ phục vụ cho cả 3 Assignment:
- Biểu đồ phân phối class
- Confusion matrix + per-class F1
- So sánh DT vs RF (NumPy)
- Confusion matrices cho sklearn
- Feature importance (sklearn)
- So sánh kết quả cuối cùng giữa 4 mô hình
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 110, 'axes.titlesize': 13})


def plot_class_distribution(
    red_df:   pd.DataFrame,
    white_df: pd.DataFrame,
    df:       pd.DataFrame,
) -> None:
    """
    Vẽ 3 biểu đồ cột: phân phối class cho Red, White và Combined.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, frame, title, color in zip(
        axes,
        [red_df, white_df, df],
        ['Red Wine', 'White Wine', 'Combined'],
        ['#c0392b', '#e0aa3e', '#2980b9'],
    ):
        counts = frame['quality'].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=color,
               edgecolor='white', linewidth=0.8)
        ax.set_title(f'{title} – Quality Distribution', fontweight='bold')
        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Count')
        ax.set_xticks(counts.index)
        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=9,
            )
    plt.suptitle('Wine Quality Score Distribution',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_confusion_and_f1(
    cm:           np.ndarray,
    f1_per_class: np.ndarray,
    f1_weighted:  float,
    classes:      list[int],
    title:        str,
    cm_cmap:      str = 'Blues',
    bar_palette:  str = 'Blues_d',
) -> None:
    """
    Vẽ 2 biểu đồ song song: confusion matrix (heatmap) và per-class F1 (bar).
    Có đường tham chiếu weighted average ở biểu đồ bar.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap=cm_cmap,
                xticklabels=classes, yticklabels=classes,
                ax=axes[0], linewidths=0.4)
    axes[0].set_title(f'Confusion Matrix – {title}', fontweight='bold')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    bars = axes[1].bar(
        [str(c) for c in classes], f1_per_class,
        color=sns.color_palette(bar_palette, len(classes)), edgecolor='white',
    )
    axes[1].axhline(f1_weighted, color='red', linestyle='--', linewidth=1.4,
                    label=f'Weighted avg = {f1_weighted:.3f}')
    axes[1].set_title(f'Per-class F1 Score – {title}', fontweight='bold')
    axes[1].set_xlabel('Quality Class')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()
    for bar, val in zip(bars, f1_per_class):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9,
        )
    plt.tight_layout()
    plt.show()


def plot_numpy_comparison(
    classes:        list[int],
    f1s_dt:         np.ndarray,
    f1s_rf:         np.ndarray,
    f1_dt_weighted: float,
    f1_rf_weighted: float,
    f1_dt_macro:    float,
    f1_rf_macro:    float,
) -> None:
    """
    So sánh DT vs RF (NumPy): per-class F1 và overall F1.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(classes))
    w = 0.35
    axes[0].bar(x - w/2, f1s_dt, w, label='Decision Tree',
                color='#3498db', edgecolor='white')
    axes[0].bar(x + w/2, f1s_rf, w, label='Random Forest',
                color='#2ecc71', edgecolor='white')
    axes[0].set_title('Per-class F1 Score: DT vs RF (NumPy)', fontweight='bold')
    axes[0].set_xlabel('Quality Class')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(c) for c in classes])
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()

    model_names = ['DT (NumPy)', 'RF (NumPy)']
    w_vals      = [f1_dt_weighted, f1_rf_weighted]
    m_vals      = [f1_dt_macro,    f1_rf_macro]
    x2 = np.arange(len(model_names))
    axes[1].bar(x2 - 0.2, w_vals, 0.35, label='Weighted F1',
                color='#3498db', edgecolor='white')
    axes[1].bar(x2 + 0.2, m_vals, 0.35, label='Macro F1',
                color='#e74c3c', edgecolor='white')
    axes[1].set_title('Overall F1 Score: DT vs RF (NumPy)', fontweight='bold')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(model_names)
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()
    for bar, val in zip(axes[1].patches, w_vals + m_vals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold',
        )
    plt.tight_layout()
    plt.show()


def plot_sklearn_confusion_matrices(
    y_test:    np.ndarray,
    y_pred_dt: np.ndarray,
    y_pred_rf: np.ndarray,
    classes:   list[int],
) -> None:
    """
    Vẽ confusion matrix song song cho DT và RF (sklearn).
    """
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, y_pred, title, cmap in zip(
        axes,
        [y_pred_dt, y_pred_rf],
        ['Decision Tree (sklearn)', 'Random Forest (sklearn)'],
        ['Oranges', 'Purples'],
    ):
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, linewidths=0.4)
        ax.set_title(f'Confusion Matrix – {title}', fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    sklearn_dt,
    sklearn_rf,
    feature_names: list[str],
) -> None:
    """
    Vẽ horizontal bar chart tầm quan trọng của các đặc trưng từ
    sklearn DT và sklearn RF.
    """
    imp_dt = sklearn_dt.feature_importances_
    imp_rf = sklearn_rf.feature_importances_
    idx_dt = np.argsort(imp_dt)
    idx_rf = np.argsort(imp_rf)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, imp, idx, title, color in zip(
        axes,
        [imp_dt, imp_rf],
        [idx_dt, idx_rf],
        ['Decision Tree', 'Random Forest'],
        ['#e67e22', '#8e44ad'],
    ):
        sorted_names = [feature_names[i] for i in idx]
        sorted_vals  = imp[idx]
        bars = ax.barh(sorted_names, sorted_vals, color=color, edgecolor='white')
        ax.set_title(f'Feature Importance – {title} (sklearn)', fontweight='bold')
        ax.set_xlabel('Importance Score')
        for bar, val in zip(bars, sorted_vals):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_final_comparison(
    f1_dt_weighted:         float,
    f1_rf_weighted:         float,
    f1_sklearn_dt_weighted: float,
    f1_sklearn_rf_weighted: float,
    f1_dt_macro:            float,
    f1_rf_macro:            float,
    f1_sklearn_dt_macro:    float,
    f1_sklearn_rf_macro:    float,
) -> None:
    """
    Vẽ biểu đồ so sánh F1 (Weighted + Macro) của cả 4 mô hình
    và in bảng tóm tắt cuối cùng.
    """
    model_names = ['DT\n(NumPy)', 'RF\n(NumPy)', 'DT\n(sklearn)', 'RF\n(sklearn)']
    w_scores    = [f1_dt_weighted, f1_rf_weighted,
                   f1_sklearn_dt_weighted, f1_sklearn_rf_weighted]
    m_scores    = [f1_dt_macro,    f1_rf_macro,
                   f1_sklearn_dt_macro,    f1_sklearn_rf_macro]
    colors      = ['#3498db', '#2ecc71', '#e67e22', '#8e44ad']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, scores, metric in zip(axes, [w_scores, m_scores],
                                   ['Weighted', 'Macro']):
        bars = ax.bar(model_names, scores, color=colors,
                      edgecolor='white', linewidth=0.8)
        ax.set_title(f'F1 Score ({metric}) – All Models', fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold',
            )
    plt.suptitle('Performance Comparison – All Models',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()

    summary = pd.DataFrame({
        'Model'      : ['DT (NumPy)', 'RF (NumPy)', 'DT (sklearn)', 'RF (sklearn)'],
        'F1 Weighted': [f'{v:.4f}' for v in w_scores],
        'F1 Macro'   : [f'{v:.4f}' for v in m_scores],
    })
    print("\n===== Final Summary =====")
    print(summary.to_string(index=False))
