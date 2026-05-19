# Wine Quality Classification

Phân loại chất lượng rượu (**quality score 3–9**) bằng Decision Tree và Random Forest.
Dataset: [Wine Quality – UCI ML Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)

---

## Cấu trúc project

```
lab4/
├── config.yaml         # Cấu hình dataset (đường dẫn tìm + URL tải)
├── __init__.py
├── data_loader.py      # Đọc config + load dataset (tìm local hoặc tải về)
├── preprocessing.py    # Stratified train/test split thuần NumPy
├── metrics.py          # F1 Score / Confusion Matrix / Classification Report
├── dt_numpy.py         # Decision Tree tự cài đặt (NumPy + Gini)        – AS1
├── rf_numpy.py         # Random Forest tự cài đặt (NumPy + Bootstrap)   – AS2
├── dt_sklearn.py       # Decision Tree dùng scikit-learn                – AS3
├── rf_sklearn.py       # Random Forest dùng scikit-learn                – AS3
├── visualization.py    # Tất cả hàm vẽ biểu đồ
├── main_as1.py         # Chạy Assignment 1
├── main_as2.py         # Chạy Assignment 2
├── main_as3.py         # Chạy Assignment 3
├── requirements.txt
├── .gitignore
└── README.md
```

> Dataset **không được commit** lên repo. Code sẽ tự tải về lần đầu chạy.

---

## Mô tả Assignment

### Assignment 1 – Decision Tree NumPy (`main_as1.py` + `dt_numpy.py`)
- Cây quyết định với **Gini impurity**.
- Tìm điểm chia tốt nhất bằng **cumulative-sum vector hoá**.
- Đánh giá tập test: **F1 Score** (weighted + macro), Confusion Matrix, Per-class F1.

### Assignment 2 – Random Forest NumPy (`main_as2.py` + `rf_numpy.py`)
- Tổ hợp **50 Decision Tree** thuần NumPy.
- Mỗi cây train trên một **bootstrap sample**, mỗi split chỉ xét `sqrt(n_features)` đặc trưng ngẫu nhiên.
- Dự đoán cuối cùng = **majority vote**.

### Assignment 3 – Sklearn DT + RF (`main_as3.py` + `dt_sklearn.py` + `rf_sklearn.py`)
- Dùng `sklearn.tree.DecisionTreeClassifier` và `sklearn.ensemble.RandomForestClassifier`.
- Trực quan **Feature Importance**.
- **So sánh** F1 Score giữa 4 mô hình (NumPy DT, NumPy RF, sklearn DT, sklearn RF).

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Cách chạy

### Workflow tự động (mặc định)

Chỉ cần chạy thẳng:

```bash
python main_as1.py
python main_as2.py
python main_as3.py
```

Lần đầu chạy, code sẽ tự tải Wine Quality CSV từ UCI repository về thư mục `./data/`.
Các lần chạy sau, code dùng lại file đã tải.

### Workflow thủ công (nếu đã có sẵn dataset)

1. Đặt 2 file CSV vào **bất kỳ thư mục nào** trong danh sách `search_paths` của `config.yaml`:

```yaml
dataset:
  search_paths:
    - .
    - ./data
    - ./dataset
    - ~/datasets/wine_quality
```

2. Chạy bình thường:

```bash
python main_as1.py
```

---

## Tùy biến qua `config.yaml`

Toàn bộ cấu hình dataset nằm tách biệt khỏi code, có thể chỉnh không cần sửa Python:

```yaml
dataset:
  files:
    red:   winequality-red.csv
    white: winequality-white.csv

  search_paths:
    - .
    - ./data

  download:
    enabled: true                                                                   # đặt false để tắt auto-download
    base_url: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
    save_to:  ./data
```

---

## Kết quả

| Model             | F1 Weighted | F1 Macro |
|-------------------|-------------|----------|
| Decision Tree (NumPy)   | —     | —    |
| Random Forest (NumPy)   | —     | —    |
| Decision Tree (sklearn) | —     | —    |
| Random Forest (sklearn) | —     | —    |

*(Cập nhật sau khi chạy thực nghiệm)*
