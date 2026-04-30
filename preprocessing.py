"""
preprocessing.py
================
Tiền xử lý dữ liệu sau khi load bằng data_loader:
  - Shuffle tập train
  - Load & tiền xử lý toàn bộ 3 split (train / val / test)

Lưu ý: data_loader.load_split() đã trả về X dạng float32, flatten,
        giá trị [0,1] — preprocessing.py chỉ cần lo việc shuffle.
"""

import numpy as np
from pathlib import Path
from data_loader import find_split_dir, load_split


def shuffle_dataset(
    X:    np.ndarray,
    y:    np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shuffle ngẫu nhiên X và y theo cùng một hoán vị.

    Parameters
    ----------
    X    : np.ndarray – mảng đặc trưng, shape (N, D)
    y    : np.ndarray – mảng nhãn, shape (N,)
    seed : int        – random seed để kết quả có thể tái lập

    Returns
    -------
    X_shuffled, y_shuffled : np.ndarray
    """
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def load_and_preprocess_all(
    base_path: str | Path,
    img_size:  int = 128,
    seed:      int = 42,
) -> tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Tìm, load và chuẩn bị cả 3 split trong một lần gọi.

    Dùng find_split_dir() để tự động định vị thư mục train/val/test
    → hoạt động đúng trên cả Kaggle lẫn máy local mà không cần
      chỉnh đường dẫn thủ công.

    Parameters
    ----------
    base_path : str | Path – thư mục gốc chứa dataset (bất kỳ tầng nào)
    img_size  : int        – kích thước resize ảnh (mặc định 128)
    seed      : int        – random seed để shuffle train có thể tái lập

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
      X : float32, shape (N, img_size²), giá trị [0, 1]
      y : int32,   shape (N,),           giá trị ±1
    """
    base = Path(base_path)

    print("=" * 45)
    print("  Đang load dữ liệu...")
    print("=" * 45)

    for split in ("train", "val", "test"):
        print(f"\n[{split}]")
        split_dir = find_split_dir(base, split)

        if split == "train":
            X_train, y_train = load_split(split_dir, img_size)
            X_train, y_train = shuffle_dataset(X_train, y_train, seed)
        elif split == "val":
            X_val, y_val = load_split(split_dir, img_size)
        else:
            X_test, y_test = load_split(split_dir, img_size)

    print("\n" + "=" * 45)
    print(f"  Train : {X_train.shape}  NORMAL={np.sum(y_train==-1)}  PNEUMONIA={np.sum(y_train==1)}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")
    print("=" * 45)

    return X_train, y_train, X_val, y_val, X_test, y_test
