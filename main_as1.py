"""
main_as1.py
-----------
Assignment 1: Soft-margin SVM tự cài đặt bằng NumPy + SGD.

Pipeline:
  1. Load & tiền xử lý dữ liệu
  2. Huấn luyện Custom SVM (có theo dõi Validation F1)
  3. Đánh giá trên tập test (Precision, Recall, F1)
  4. Vẽ biểu đồ: Training curve, Confusion matrix, Metrics bar

"""

from pathlib import Path
from preprocessing import load_and_preprocess_all
from svm_numpy     import SVM
from metrics       import compute_metrics, print_metrics
from visualization import (
    show_samples,
    plot_training_curve,
    plot_confusion_matrix,
    plot_metrics_bar,
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

IMG_SIZE   = 128
SVM_CONFIG = dict(lr=0.001, lambda_param=0.01, n_iters=20)

# ------------------------------------------------------------------ #
#  1. Load & tiền xử lý                                               #
# ------------------------------------------------------------------ #

print("=" * 55)
print("  ASSIGNMENT 1 – Custom SVM (NumPy + SGD)")
print("=" * 55)

X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_all(
    BASE_PATH, img_size=IMG_SIZE
)

# Hiển thị ảnh mẫu
show_samples(X_train, y_train, img_size=IMG_SIZE, n=6, title="Sample Training Images")

# ------------------------------------------------------------------ #
#  2. Huấn luyện Custom SVM                                           #
# ------------------------------------------------------------------ #

print("\nHuấn luyện Custom SVM...")
model = SVM(**SVM_CONFIG)
model.fit(X_train, y_train, X_val, y_val)

# ------------------------------------------------------------------ #
#  3. Dự đoán & Đánh giá                                              #
# ------------------------------------------------------------------ #

y_pred = model.predict(X_test)
precision, recall, f1, cm = compute_metrics(y_test, y_pred)
print_metrics(precision, recall, f1, cm, model_name="Custom SVM – Test Set")

# ------------------------------------------------------------------ #
#  4. Vẽ biểu đồ                                                      #
# ------------------------------------------------------------------ #

plot_training_curve(
    model.loss_history,
    model.val_f1_history,
    title="Custom SVM – Training Loss & Validation F1",
)

plot_confusion_matrix(cm, title="Custom SVM – Confusion Matrix", cmap="Blues")

plot_metrics_bar(precision, recall, f1, title="Custom SVM – Metrics", color="steelblue")
