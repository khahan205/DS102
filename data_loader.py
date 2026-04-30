"""
data_loader.py
==============
Hàm tiện ích để tìm và load dữ liệu ảnh Chest X-Ray (NORMAL / PNEUMONIA).
"""

import numpy as np
from pathlib import Path
from PIL import Image


def find_split_dir(base: Path, split: str) -> Path:
    """
    Tìm đệ quy thư mục train / val / test dù dataset nằm ở tầng nào.

    Parameters
    ----------
    base  : Path – thư mục gốc của dataset (truyền vào bất kỳ tầng nào cũng được)
    split : str  – tên split cần tìm, ví dụ "train", "val", "test"

    Returns
    -------
    Path – đường dẫn đến thư mục split chứa NORMAL / PNEUMONIA

    Raises
    ------
    FileNotFoundError – nếu không tìm thấy split thoả điều kiện
    """
    for p in base.rglob(split):
        if p.is_dir():
            children = [c.name for c in p.iterdir() if c.is_dir()]
            if any(c in children for c in ["NORMAL", "PNEUMONIA"]):
                return p
    raise FileNotFoundError(f"Cannot find '{split}' split under {base}")


def load_split(
    split_dir: Path,
    img_size:  int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load tất cả ảnh từ split_dir/{NORMAL,PNEUMONIA}, resize và flatten.

    Parameters
    ----------
    split_dir : Path – thư mục chứa 2 thư mục con NORMAL và PNEUMONIA
    img_size  : int  – kích thước ảnh sau khi resize (mặc định 128)

    Returns
    -------
    X : np.ndarray, shape (N, img_size²), dtype float32, giá trị trong [0, 1]
    y : np.ndarray, shape (N,),           dtype int32
        NORMAL = -1, PNEUMONIA = +1
    """
    label_map = {"NORMAL": -1, "PNEUMONIA": 1}
    images, labels = [], []

    for cls, lbl in label_map.items():
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(cls_dir)

        files = sorted(cls_dir.glob("*.*"))
        count_before = len(labels)

        for fpath in files:
            if fpath.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
                continue
            try:
                img = (
                    Image.open(fpath)
                    .convert("L")                              # chuyển sang grayscale
                    .resize((img_size, img_size), Image.Resampling.BILINEAR)
                )
                images.append(np.array(img, dtype=np.float32).flatten() / 255.0)
                labels.append(lbl)
            except Exception as e:
                print(f"  skip {fpath.name}: {e}")

        count_loaded = len(labels) - count_before
        print(f"  {split_dir.name}/{cls}: {count_loaded} imgs")

    return np.stack(images), np.array(labels, dtype=np.int32)
