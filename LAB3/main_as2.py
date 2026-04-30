"""
main_as2.py
-----------
Assignment 2: SVM dùng thư viện scikit-learn + so sánh với Custom SVM.

Pipeline:
  1. Load & tiền xử lý dữ liệu
  2. Chạy lại Custom SVM để lấy kết quả so sánh
  3. Grid search tham số C trên Validation set
  4. Huấn luyện Sklearn SVM với C tốt nhất
  5. Đánh giá trên tập test (Precision, Recall, F1)
  6. So sánh kết quả hai mô hình

Cách chạy:
    python main_as2.py
"""

from pathlib import Path
from preprocessing import load_and_preprocess_all
from svm_numpy     import SVM
from svm_sklearn   import select_best_C, train_sklearn_svm
from metrics       import compute_metrics, print_metrics
from visualization import (
    plot_confusion_matrix,
    plot_metrics_bar,
    plot_comparison,
)

# ------------------------------------------------------------------ #
#  Cấu hình                                                           #
# ------------------------------------------------------------------ #

# Tự động detect môi trường — không cần chỉnh tay
_KAGGLE = Path("/kaggle/input")
if _KAGGLE.exists():
    BASE_PATH = _KAGGLE          # Kaggle: find_split_dir tự tìm sâu xuống
else:
    BASE_PATH = Path(__file__).parent / "data"   # Local: để dataset vào thư mục data/

IMG_SIZE = 128
C_VALUES = [0.01, 0.1, 1, 10]
KERNEL   = "linear"

# ------------------------------------------------------------------ #
#  1. Load & tiền xử lý                                               #
# ------------------------------------------------------------------ #

print("=" * 55)
print("  ASSIGNMENT 2 – Sklearn SVM + Comparison")
print("=" * 55)

X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_all(
    BASE_PATH, img_size=IMG_SIZE
)

# ------------------------------------------------------------------ #
#  2. Chạy lại Custom SVM để lấy kết quả so sánh                     #
# ------------------------------------------------------------------ #

print("\n[1/2] Huấn luyện Custom SVM (để so sánh)...")
custom_svm = SVM(lr=0.001, lambda_param=0.01, n_iters=20)
custom_svm.fit(X_train, y_train, X_val, y_val)
y_pred_custom = custom_svm.predict(X_test)
prec1, rec1, f1_1, cm1 = compute_metrics(y_test, y_pred_custom)
print_metrics(prec1, rec1, f1_1, cm1, model_name="Custom SVM – Test Set")

# ------------------------------------------------------------------ #
#  3. Grid search C & huấn luyện Sklearn SVM                         #
# ------------------------------------------------------------------ #

print("\n[2/2] Grid search C cho Sklearn SVM...")
best_C = select_best_C(X_train, y_train, X_val, y_val, C_values=C_VALUES, kernel=KERNEL)

sklearn_svm    = train_sklearn_svm(X_train, y_train, C=best_C, kernel=KERNEL)
y_pred_sklearn = sklearn_svm.predict(X_test)

# ------------------------------------------------------------------ #
#  4. Đánh giá Sklearn SVM                                            #
# ------------------------------------------------------------------ #

prec2, rec2, f1_2, cm2 = compute_metrics(y_test, y_pred_sklearn)
print_metrics(prec2, rec2, f1_2, cm2, model_name=f"Sklearn SVM (C={best_C}) – Test Set")

plot_confusion_matrix(cm2, title="Sklearn SVM – Confusion Matrix", cmap="Greens")
plot_metrics_bar(prec2, rec2, f1_2, title="Sklearn SVM – Metrics", color="tomato")

# ------------------------------------------------------------------ #
#  5. So sánh Custom vs Sklearn                                       #
# ------------------------------------------------------------------ #

print("\n" + "=" * 55)
print("  COMPARISON SUMMARY")
print("=" * 55)
print(f"  {'Metric':<12}  {'Custom SVM':>12}  {'Sklearn SVM':>12}")
print("  " + "-" * 40)
print(f"  {'Precision':<12}  {prec1:>12.4f}  {prec2:>12.4f}")
print(f"  {'Recall':<12}  {rec1:>12.4f}  {rec2:>12.4f}")
print(f"  {'F1-score':<12}  {f1_1:>12.4f}  {f1_2:>12.4f}")
print("=" * 55)

plot_comparison(
    metrics_custom  = [prec1, rec1, f1_1],
    metrics_sklearn = [prec2, rec2, f1_2],
    title="Custom SVM vs Sklearn SVM – Comparison",
)
