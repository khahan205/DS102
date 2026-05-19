"""
main_as3.py
===========
Assignment 3 — Decision Tree & Random Forest using scikit-learn.
Huấn luyện và đánh giá DT + RF bằng scikit-learn trên Wine Quality dataset,
sau đó so sánh với kết quả từ Assignment 1 + 2.
"""

import sys
import numpy as np

sys.setrecursionlimit(10000)

from data_loader   import load_wine_data, extract_features
from preprocessing import numpy_train_test_split
from dt_numpy      import DecisionTree
from rf_numpy      import RandomForest
from metrics       import numpy_f1_score
from dt_sklearn    import (train_sklearn_decision_tree,
                            evaluate_sklearn_decision_tree)
from rf_sklearn    import (train_sklearn_random_forest,
                            evaluate_sklearn_random_forest)
from visualization import (plot_sklearn_confusion_matrices,
                            plot_feature_importance,
                            plot_final_comparison)


CONFIG_PATH = "config.yaml"


def main():
    print("=" * 60)
    print("  Assignment 3: Decision Tree & Random Forest (sklearn)")
    print("=" * 60)

    _, _, df              = load_wine_data(CONFIG_PATH)
    X, y, feature_names   = extract_features(df)
    X_train, X_test, y_train, y_test = numpy_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    classes = sorted(np.unique(y).tolist())

    print(f"\nTrain : {X_train.shape[0]} samples")
    print(f"Test  : {X_test.shape[0]} samples")

    print("\n----- Re-running NumPy models for comparison -----")
    dt_np = DecisionTree(max_depth=10, min_samples_split=5, random_state=42)
    dt_np.fit(X_train, y_train)
    y_pred_dt_np   = dt_np.predict(X_test)
    f1_dt_weighted = numpy_f1_score(y_test, y_pred_dt_np, average='weighted')
    f1_dt_macro    = numpy_f1_score(y_test, y_pred_dt_np, average='macro')

    rf_np = RandomForest(
        n_estimators=50, max_depth=8, min_samples_split=5,
        max_features='sqrt', random_state=42,
    )
    rf_np.fit(X_train, y_train)
    y_pred_rf_np   = rf_np.predict(X_test)
    f1_rf_weighted = numpy_f1_score(y_test, y_pred_rf_np, average='weighted')
    f1_rf_macro    = numpy_f1_score(y_test, y_pred_rf_np, average='macro')

    print("\n----- Training sklearn models -----")
    sklearn_dt = train_sklearn_decision_tree(X_train, y_train)
    res_dt     = evaluate_sklearn_decision_tree(
        sklearn_dt, X_test, y_test, classes
    )

    sklearn_rf = train_sklearn_random_forest(X_train, y_train)
    res_rf     = evaluate_sklearn_random_forest(
        sklearn_rf, X_test, y_test, classes
    )

    plot_sklearn_confusion_matrices(
        y_test, res_dt['y_pred'], res_rf['y_pred'], classes
    )
    plot_feature_importance(sklearn_dt, sklearn_rf, feature_names)

    plot_final_comparison(
        f1_dt_weighted,        f1_rf_weighted,
        res_dt['f1_weighted'], res_rf['f1_weighted'],
        f1_dt_macro,           f1_rf_macro,
        res_dt['f1_macro'],    res_rf['f1_macro'],
    )


if __name__ == '__main__':
    main()
