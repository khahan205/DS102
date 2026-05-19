"""
dt_sklearn.py
=============
Decision Tree dùng scikit-learn — Assignment 3 (phần 1).
"""

import numpy as np
from sklearn.tree    import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def train_sklearn_decision_tree(
    X_train:           np.ndarray,
    y_train:           np.ndarray,
    max_depth:         int = 10,
    min_samples_split: int = 5,
    random_state:      int = 42,
) -> DecisionTreeClassifier:
    """
    Huấn luyện Decision Tree bằng sklearn (criterion = Gini).

    Parameters
    ----------
    X_train           : np.ndarray, shape (N, D)
    y_train           : np.ndarray, shape (N,)
    max_depth         : int – độ sâu tối đa
    min_samples_split : int – số mẫu tối thiểu để chia
    random_state      : int – seed

    Returns
    -------
    model : DecisionTreeClassifier (đã fit)
    """
    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_sklearn_decision_tree(
    model:   DecisionTreeClassifier,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    classes: list[int],
) -> dict:
    """
    Đánh giá Decision Tree (sklearn) trên tập test bằng F1 Score.

    Parameters
    ----------
    model   : DecisionTreeClassifier đã fit
    X_test  : np.ndarray, shape (N, D)
    y_test  : np.ndarray, shape (N,)
    classes : list[int] – danh sách class

    Returns
    -------
    dict gồm: y_pred, f1_weighted, f1_macro, f1_per_class, cm
    """
    y_pred       = model.predict(X_test)
    f1_weighted  = f1_score(y_test, y_pred, average='weighted')
    f1_macro     = f1_score(y_test, y_pred, average='macro')
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=classes)
    cm           = confusion_matrix(y_test, y_pred, labels=classes)
    report       = classification_report(
        y_test, y_pred, target_names=[f'Q{c}' for c in classes]
    )

    print("===== Decision Tree (sklearn) – F1 Score =====")
    print(f"F1 (weighted) : {f1_weighted:.4f}")
    print(f"F1 (macro)    : {f1_macro:.4f}")
    print()
    print(report)

    return {
        'y_pred'      : y_pred,
        'f1_weighted' : f1_weighted,
        'f1_macro'    : f1_macro,
        'f1_per_class': f1_per_class,
        'cm'          : cm,
    }
