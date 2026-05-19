"""
main_as2.py
===========
Assignment 2 — Random Forest using NumPy.
Huấn luyện và đánh giá Random Forest thuần NumPy trên Wine Quality dataset
với metric F1 Score. So sánh kết quả với Decision Tree từ Assignment 1.
"""

import sys
import numpy as np

sys.setrecursionlimit(10000)

from data_loader   import load_wine_data, extract_features
from preprocessing import numpy_train_test_split
from dt_numpy      import DecisionTree
from rf_numpy      import RandomForest
from metrics       import (numpy_f1_score, numpy_confusion_matrix,
                            numpy_classification_report)
from visualization import plot_confusion_and_f1, plot_numpy_comparison


CONFIG_PATH = "config.yaml"


def main():
    print("=" * 60)
    print("  Assignment 2: Random Forest using NumPy")
    print("=" * 60)

    _, _, df = load_wine_data(CONFIG_PATH)
    X, y, _  = extract_features(df)
    X_train, X_test, y_train, y_test = numpy_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    classes = sorted(np.unique(y).tolist())

    print(f"\nTrain : {X_train.shape[0]} samples")
    print(f"Test  : {X_test.shape[0]} samples")

    dt = DecisionTree(max_depth=10, min_samples_split=5, random_state=42)
    print("\nTraining Decision Tree (baseline) ...")
    dt.fit(X_train, y_train)
    y_pred_dt      = dt.predict(X_test)
    f1_dt_weighted = numpy_f1_score(y_test, y_pred_dt, average='weighted')
    f1_dt_macro    = numpy_f1_score(y_test, y_pred_dt, average='macro')
    f1s_dt         = numpy_f1_score(y_test, y_pred_dt, average=None)
    print(f"DT F1 (weighted) : {f1_dt_weighted:.4f}")

    rf = RandomForest(
        n_estimators=50, max_depth=8, min_samples_split=5,
        max_features='sqrt', random_state=42,
    )
    print("\nTraining Random Forest (NumPy) ...")
    rf.fit(X_train, y_train)
    print("Training complete.")

    y_pred_rf      = rf.predict(X_test)
    f1_rf_weighted = numpy_f1_score(y_test, y_pred_rf, average='weighted')
    f1_rf_macro    = numpy_f1_score(y_test, y_pred_rf, average='macro')
    f1s_rf         = numpy_f1_score(y_test, y_pred_rf, average=None)
    cm_rf          = numpy_confusion_matrix(y_test, y_pred_rf, classes)

    print("\n===== Random Forest (NumPy) – F1 Score =====")
    print(f"F1 (weighted) : {f1_rf_weighted:.4f}")
    print(f"F1 (macro)    : {f1_rf_macro:.4f}")
    print()
    numpy_classification_report(y_test, y_pred_rf, classes)

    plot_confusion_and_f1(
        cm_rf, f1s_rf, f1_rf_weighted, classes,
        title='Random Forest (NumPy)',
        cm_cmap='Greens', bar_palette='Greens_d',
    )
    plot_numpy_comparison(
        classes, f1s_dt, f1s_rf,
        f1_dt_weighted, f1_rf_weighted,
        f1_dt_macro,    f1_rf_macro,
    )


if __name__ == '__main__':
    main()
