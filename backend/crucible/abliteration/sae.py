from __future__ import annotations
# Sparse autoencoder (dictionary learning) — the current state of the art for turning a
# model's dense, polysemantic activations into a large set of sparse, more-monosemantic
# features you can name and target. An overcomplete encoder (d -> m, m > d) with a ReLU and
# an L1 sparsity penalty learns a dictionary: each activation is reconstructed as a sparse
# sum of a few learned feature directions. Pure numpy + manual backprop, so the learning is
# deterministic and unit-tested on synthetic sparse data (it recovers the planted features).
import numpy as np
from numpy.typing import ArrayLike


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class SparseAutoencoder:
    def __init__(self, n_features: int, l1: float = 1e-2, lr: float = 1e-2,
                 epochs: int = 300, seed: int = 0):
        self.m = int(n_features)
        self.l1 = l1
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.W_e: np.ndarray | None = None
        self.b_e: np.ndarray | None = None
        self.W_d: np.ndarray | None = None
        self.b_d: np.ndarray | None = None
        self.history: list[float] = []

    def fit(self, X: ArrayLike) -> "SparseAutoencoder":
        Xa = np.asarray(X, dtype=np.float64)
        n, d = Xa.shape
        rng = np.random.default_rng(self.seed)
        self.W_e = rng.standard_normal((d, self.m)) * (1.0 / np.sqrt(d))
        self.b_e = np.zeros(self.m)
        self.W_d = rng.standard_normal((self.m, d)) * (1.0 / np.sqrt(self.m))
        self.b_d = np.zeros(d)
        for _ in range(self.epochs):
            pre = Xa @ self.W_e + self.b_e        # (n, m)
            z = _relu(pre)
            xhat = z @ self.W_d + self.b_d        # (n, d)
            resid = xhat - Xa
            mse = float((resid ** 2).mean())
            self.history.append(mse)
            # grads
            dxhat = (2.0 / n) * resid             # (n, d)
            dW_d = z.T @ dxhat + 0.0
            db_d = dxhat.sum(axis=0)
            dz = dxhat @ self.W_d.T               # (n, m)
            dz += (self.l1 / n) * np.sign(z)      # L1 on the codes
            dpre = dz * (pre > 0)                 # ReLU grad
            dW_e = Xa.T @ dpre
            db_e = dpre.sum(axis=0)
            self.W_d -= self.lr * dW_d
            self.b_d -= self.lr * db_d
            self.W_e -= self.lr * dW_e
            self.b_e -= self.lr * db_e
            # keep decoder atoms unit-norm (standard SAE constraint; prevents collapse)
            norms = np.linalg.norm(self.W_d, axis=1, keepdims=True) + 1e-8
            self.W_d = self.W_d / norms
        return self

    def encode(self, X: ArrayLike) -> np.ndarray:
        if self.W_e is None:
            raise RuntimeError("SAE not fitted")
        return _relu(np.asarray(X, dtype=np.float64) @ self.W_e + self.b_e)

    def decode(self, Z: ArrayLike) -> np.ndarray:
        if self.W_d is None:
            raise RuntimeError("SAE not fitted")
        return np.asarray(Z, dtype=np.float64) @ self.W_d + self.b_d

    def reconstruct(self, X: ArrayLike) -> np.ndarray:
        return self.decode(self.encode(X))

    def reconstruction_error(self, X: ArrayLike) -> float:
        Xa = np.asarray(X, dtype=np.float64)
        return float(((self.reconstruct(Xa) - Xa) ** 2).mean())

    def r2(self, X: ArrayLike) -> float:
        Xa = np.asarray(X, dtype=np.float64)
        ss_res = float(((self.reconstruct(Xa) - Xa) ** 2).sum())
        ss_tot = float(((Xa - Xa.mean(axis=0)) ** 2).sum()) + 1e-12
        return 1.0 - ss_res / ss_tot

    def sparsity(self, X: ArrayLike, eps: float = 1e-6) -> float:
        """Mean fraction of features that are ~zero for a given activation (higher = sparser)."""
        z = self.encode(X)
        return float((z <= eps).mean())

    def feature_directions(self) -> np.ndarray:
        """The learned dictionary atoms (unit-norm decoder rows) — each a candidate
        monosemantic feature direction in activation space."""
        if self.W_d is None:
            raise RuntimeError("SAE not fitted")
        return self.W_d.copy()

    def top_features(self, x: ArrayLike, k: int = 8) -> list[tuple[int, float]]:
        """The k strongest-firing features for a single activation (feature index, value)."""
        z = self.encode(np.asarray(x, dtype=np.float64)[None, :])[0]
        idx = np.argsort(z)[::-1][:k]
        return [(int(i), float(z[i])) for i in idx if z[i] > 0]


def label_features(sae: SparseAutoencoder, X: ArrayLike, tokens: list[str],
                   n_features: int = 12, n_tokens: int = 6) -> list[dict]:
    """Monosemanticity readout: for the most-used features, the tokens that fire them
    hardest. A feature that fires on a coherent token set ('refuse', 'cannot', 'sorry') is
    interpretable; one that fires on noise isn't. tokens[i] labels row i of X."""
    Xa = np.asarray(X, dtype=np.float64)
    Z = sae.encode(Xa)                          # (n, m)
    usage = Z.sum(axis=0)
    top = np.argsort(usage)[::-1][:n_features]
    out: list[dict] = []
    for f in top:
        col = Z[:, int(f)]
        order = np.argsort(col)[::-1]
        fires = []
        for i in order[:n_tokens]:
            if col[i] > 0:
                fires.append(tokens[int(i)])
        if fires:
            out.append({"feature": int(f), "usage": float(usage[int(f)]),
                        "peak": float(col[order[0]]), "fires_on": fires})
    return out
