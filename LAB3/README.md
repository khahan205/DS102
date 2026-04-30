# SVM Pneumonia Detection

Phân loại ảnh X-Ray phổi (**NORMAL / PNEUMONIA**) bằng SVM.  
Dataset: [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## Cấu trúc project

```
svm_pneumonia/
├── __init__.py         # Package init, export các module
├── data_loader.py      # Đọc ảnh thô từ disk
├── preprocessing.py    # Flatten, chuẩn hoá, shuffle
├── svm_numpy.py        # Soft-margin SVM tự cài đặt (NumPy + SGD)
├── svm_sklearn.py      # SVM dùng scikit-learn + grid search C
├── metrics.py          # Precision / Recall / F1 / Confusion Matrix
├── visualization.py    # Tất cả hàm vẽ biểu đồ
├── main_as1.py         # Chạy Assignment 1 (Custom SVM)
├── main_as2.py         # Chạy Assignment 2 (Sklearn SVM + so sánh)
├── requirements.txt
└── README.md
```

---

## Mô tả Assignment

### Assignment 1 – Custom SVM (`main_as1.py`)
- Soft-margin SVM với **hinge loss** + **L2 regularisation**.
- Tối ưu bằng **SGD** (Stochastic Gradient Descent).
- Theo dõi **Training Loss** và **Validation F1** sau mỗi epoch.
- Đánh giá tập test: **Precision, Recall, F1-score**, Confusion Matrix.

### Assignment 2 – Sklearn SVM (`main_as2.py`)
- Dùng `sklearn.svm.SVC` với kernel tuyến tính.
- **Grid search** `C` ∈ {0.01, 0.1, 1, 10} trên Validation F1.
- Huấn luyện mô hình cuối với C tốt nhất.
- **So sánh** kết quả với Custom SVM.

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Cách chạy

1. Chỉnh `BASE_PATH` trong `main_as1.py` / `main_as2.py`:

```python
BASE_PATH = "/path/to/chest_xray"
```

2. Chạy từng Assignment:

```bash
# Assignment 1
python main_as1.py

# Assignment 2
python main_as2.py
```

---

## Kết quả

| Model       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| Custom SVM  | —         | —      | —        |
| Sklearn SVM | —         | —      | —        |

*(Cập nhật sau khi chạy thực nghiệm)*
