"""
Wine Quality Classification — LAB4
Dataset: Wine Quality (Red + White) - UCI ML Repository
"""

from .data_loader   import load_wine_data, extract_features
from .preprocessing import numpy_train_test_split
from .dt_numpy      import Node, DecisionTree
from .rf_numpy      import RandomForest
from .dt_sklearn    import (
    train_sklearn_decision_tree,
    evaluate_sklearn_decision_tree,
)
from .rf_sklearn    import (
    train_sklearn_random_forest,
    evaluate_sklearn_random_forest,
)
from .metrics import (
    numpy_f1_score,
    numpy_confusion_matrix,
    numpy_classification_report,
)
from .visualization import (
    plot_class_distribution,
    plot_confusion_and_f1,
    plot_numpy_comparison,
    plot_sklearn_confusion_matrices,
    plot_feature_importance,
    plot_final_comparison,
)
