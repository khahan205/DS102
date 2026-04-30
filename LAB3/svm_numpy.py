"""
svm_numpy.py
------------
Soft-margin SVM tự cài đặt bằng NumPy thuần, tối ưu bằng SGD.

Hàm mất mát:
    L(w, b) = λ · ||w||² + (1/N) · Σ max(0, 1 − yᵢ(w·xᵢ + b))

Gradient:
    - Nếu yᵢ(w·xᵢ + b) ≥ 1  →  ∂L/∂w = 2λw,        ∂L/∂b = 0
    - Ngược lại               →  ∂L/∂w = 2λw − yᵢxᵢ,  ∂L/∂b = −yᵢ
"""

import numpy as np
from sklearn.metrics import f1_score


class SVM:
    """
    Soft-margin SVM với SGD.

    Parameters
    ----------
    lr           : float  – Learning rate (mặc định 0.001).
    lambda_param : float  – Hệ số regularisation L2 (mặc định 0.01).
    n_iters      : int    – Số epoch huấn luyện (mặc định 20).

    Attributes (sau khi gọi fit)
    ----------------------------
    w                : np.ndarray – Vector trọng số, shape (D,).
    b                : float      – Bias.
    loss_history     : list[float] – Hinge loss sau mỗi epoch.
    val_f1_history   : list[float] – F1 trên val sau mỗi epoch (nếu truyền val).
    """

    def __init__(
        self,
        lr:           float = 0.001,
        lambda_param: float = 0.01,
        n_iters:      int   = 20,
    ):
        self.lr           = lr
        self.lambda_param = lambda_param
        self.n_iters      = n_iters

        self.w              = None
        self.b              = 0.0
        self.loss_history   = []
        self.val_f1_history = []

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _hinge_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Tính hinge loss + L2 regularisation."""
        margins = np.maximum(0, 1 - y * (X @ self.w + self.b))
        return self.lambda_param * float(self.w @ self.w) + float(np.mean(margins))

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X:     np.ndarray,
        y:     np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> "SVM":
        """
        Huấn luyện SVM bằng Stochastic Gradient Descent.

        Args:
            X     : Đặc trưng tập train, shape (N, D).
            y     : Nhãn tập train, giá trị ±1.
            X_val : Đặc trưng tập validation (tuỳ chọn).
            y_val : Nhãn tập validation (tuỳ chọn).

        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.w              = np.zeros(n_features)
        self.b              = 0.0
        self.loss_history   = []
        self.val_f1_history = []

        for epoch in range(self.n_iters):
            # Duyệt từng mẫu (stochastic)
            for idx in range(n_samples):
                x_i = X[idx]
                y_i = y[idx]

                if y_i * (x_i @ self.w + self.b) >= 1:
                    # Không vi phạm lề → chỉ cập nhật regularisation
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Vi phạm lề → cập nhật cả w và b
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_i * x_i)
                    self.b += self.lr * y_i

            # Ghi lại loss
            loss = self._hinge_loss(X, y)
            self.loss_history.append(loss)

            # Ghi lại Validation F1
            if X_val is not None and y_val is not None:
                y_pred_val = self.predict(X_val)
                val_f1 = f1_score(
                    (y_val      == 1).astype(int),
                    (y_pred_val == 1).astype(int),
                    zero_division=0,
                )
                self.val_f1_history.append(val_f1)
                print(f"  Epoch [{epoch+1:3d}/{self.n_iters}]  loss={loss:.4f}  val_F1={val_f1:.4f}")
            else:
                print(f"  Epoch [{epoch+1:3d}/{self.n_iters}]  loss={loss:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán nhãn cho X.

        Args:
            X : Shape (M, D).

        Returns:
            Nhãn dự đoán ±1, shape (M,).
        """
        return np.sign(X @ self.w + self.b)
