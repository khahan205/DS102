"""
main_as1.py
===========
Assignment 1 — Decision Tree using NumPy.
Huấn luyện và đánh giá Decision Tree thuần NumPy trên Wine Quality dataset
với metric F1 Score.
"""

import sys
import numpy as np

sys.setrecursionlimit(10000)

from data_loader   import load_wine_data, extract_features
from preprocessing import numpy_train_test_split
from dt_numpy      import DecisionTree
from metrics       import (numpy_f1_score, numpy_confusion_matrix,
                            numpy_classification_report)
from visualization import plot_class_distribution, plot_confusion_and_f1


CONFIG_PATH = "config.yaml"


def main():
    print("=" * 60)
    print("  Assignment 1: Decision Tree using NumPy")
    print("=" * 60)

    red_df, white_df, df = load_wine_data(CONFIG_PATH)
    plot_class_distribution(red_df, white_df, df)

    X, y, _ = extract_features(df)
    X_train, X_test, y_train, y_test = numpy_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    classes = sorted(np.unique(y).tolist())

    print(f"\nTrain : {X_train.shape[0]} samples")
    print(f"Test  : {X_test.shape[0]} samples")
    print(f"Features : {X_train.shape[1]}")
    print(f"Classes  : {classes}")

    dt = DecisionTree(max_depth=10, min_samples_split=5, random_state=42)
    print("\nTraining Decision Tree (NumPy) ...")
    dt.fit(X_train, y_train)
    print("Training complete.")

    y_pred         = dt.predict(X_test)
    f1_weighted    = numpy_f1_score(y_test, y_pred, average='weighted')
    f1_macro       = numpy_f1_score(y_test, y_pred, average='macro')
    f1_per_class   = numpy_f1_score(y_test, y_pred, average=None)
    cm             = numpy_confusion_matrix(y_test, y_pred, classes)

    print("\n===== Decision Tree (NumPy) – F1 Score =====")
    print(f"F1 (weighted) : {f1_weighted:.4f}")
    print(f"F1 (macro)    : {f1_macro:.4f}")
    print()
    numpy_classification_report(y_test, y_pred, classes)

    plot_confusion_and_f1(
        cm, f1_per_class, f1_weighted, classes,
        title='Decision Tree (NumPy)',
        cm_cmap='Blues', bar_palette='Blues_d',
    )


if __name__ == '__main__':
    main()
