from __future__ import annotations
# Trained refusal classifier. String matching ("I'm sorry", "I can't") is brittle: it
# misses paraphrased refusals and false-fires on benign apologies. This is a small logistic
# regression over the model's own activations (or any feature vector) — labels: 1 = refusal,
# 0 = compliance. It learns the linear refusal boundary directly. Pure numpy, deterministic.
import numpy as np
from numpy.typing import ArrayLike


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class RefusalClassifier:
    def __init__(self, lr: float = 0.1, epochs: int = 400, l2: float = 1e-3):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RefusalClassifier":
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64).ravel()
        self.mu = Xa.mean(axis=0)
        self.sd = Xa.std(axis=0) + 1e-8
        Xs = (Xa - self.mu) / self.sd
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = _sigmoid(Xs @ self.w + self.b)
            err = p - ya
            self.w -= self.lr * (Xs.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * float(err.mean())
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if self.w is None or self.mu is None or self.sd is None:
            raise RuntimeError("classifier not fitted")
        Xs = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        return _sigmoid(Xs @ self.w + self.b)

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X: ArrayLike, y: ArrayLike) -> float:
        return float((self.predict(X) == np.asarray(y).ravel().astype(int)).mean())
