"""
rf_sklearn.py
=============
Random Forest dùng scikit-learn — Assignment 3 (phần 2).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import f1_score, classification_report, confusion_matrix


def train_sklearn_random_forest(
    X_train:           np.ndarray,
    y_train:           np.ndarray,
    n_estimators:      int = 100,
    max_features:      str = 'sqrt',
    min_samples_split: int = 5,
    random_state:      int = 42,
) -> RandomForestClassifier:
    """
    Huấn luyện Random Forest bằng sklearn.

    Parameters
    ----------
    X_train           : np.ndarray, shape (N, D)
    y_train           : np.ndarray, shape (N,)
    n_estimators      : int – số cây
    max_features      : str – chiến lược chọn đặc trưng ('sqrt' / 'log2')
    min_samples_split : int – số mẫu tối thiểu để chia
    random_state      : int – seed

    Returns
    -------
    model : RandomForestClassifier (đã fit)
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        max_features=max_features,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_sklearn_random_forest(
    model:   RandomForestClassifier,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    classes: list[int],
) -> dict:
    """
    Đánh giá Random Forest (sklearn) trên tập test bằng F1 Score.

    Parameters
    ----------
    model   : RandomForestClassifier đã fit
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

    print("===== Random Forest (sklearn) – F1 Score =====")
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
