"""
SVM Pneumonia Detection — LAB3
Dataset: Chest X-Ray Images (Pneumonia) - Kaggle
"""

from .data_loader   import find_split_dir, load_split
from .preprocessing import shuffle_dataset, load_and_preprocess_all
from .svm_numpy     import SVM
from .svm_sklearn   import select_best_C, train_sklearn_svm
from .metrics       import compute_metrics, print_metrics
from .visualization import (
    show_samples,
    plot_training_curve,
    plot_confusion_matrix,
    plot_metrics_bar,
    plot_comparison,
)
