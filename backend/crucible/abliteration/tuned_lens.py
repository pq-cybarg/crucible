from __future__ import annotations
# Tuned lens. The "logit lens" reads intermediate layers by pushing their raw residual
# straight through the final unembedding — but early layers live in a different basis, so it
# reads as noise. A tuned lens learns a small affine map per layer (a "translator") that
# aligns each layer's residual to the final-layer residual before decoding, so intermediate
# predictions become faithful. It also gives a clean per-layer "decodability" curve: where
# the model's final answer becomes predictable = where it commits. Closed-form least squares,
# so it's deterministic and unit-tested.
import numpy as np
from numpy.typing import ArrayLike


def fit_affine(X: ArrayLike, Y: ArrayLike, ridge: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine map X -> Y: (A, b) minimizing ||X A + b - Y||^2 with ridge."""
    Xa = np.asarray(X, dtype=np.float64)
    Ya = np.asarray(Y, dtype=np.float64)
    n, d = Xa.shape
    aug = np.hstack([Xa, np.ones((n, 1))])          # bias column
    G = aug.T @ aug + ridge * np.eye(d + 1)
    W = np.linalg.solve(G, aug.T @ Ya)              # (d+1, out)
    return W[:-1], W[-1]


class TunedLens:
    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge
        self.maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, layer_acts: dict[int, np.ndarray], final: ArrayLike) -> "TunedLens":
        """Learn one translator per layer, mapping that layer's residuals to the final
        residual (last-token states aligned by row)."""
        Yf = np.asarray(final, dtype=np.float64)
        for layer, H in layer_acts.items():
            self.maps[int(layer)] = fit_affine(H, Yf, self.ridge)
        return self

    def translate(self, layer: int, H: ArrayLike) -> np.ndarray:
        A, b = self.maps[int(layer)]
        return np.asarray(H, dtype=np.float64) @ A + b

    def residual(self, layer: int, H: ArrayLike, final: ArrayLike) -> float:
        """Mean squared error of the tuned prediction vs the true final residual."""
        pred = self.translate(layer, H)
        return float(((pred - np.asarray(final, dtype=np.float64)) ** 2).mean())

    def raw_residual(self, H: ArrayLike, final: ArrayLike) -> float:
        """MSE of the RAW residual vs final (the logit-lens baseline: no translator)."""
        return float(((np.asarray(H, dtype=np.float64) - np.asarray(final, dtype=np.float64)) ** 2).mean())

    def decodability(self, layer: int, H: ArrayLike, final: ArrayLike) -> float:
        """1 - tuned_residual / total_variance: how well this layer predicts the final state
        (0 = no better than the mean, 1 = perfect). The per-layer 'commitment' curve."""
        Yf = np.asarray(final, dtype=np.float64)
        pred = self.translate(layer, H)
        ss_res = float(((pred - Yf) ** 2).sum())
        ss_tot = float(((Yf - Yf.mean(axis=0)) ** 2).sum()) + 1e-12
        return 1.0 - ss_res / ss_tot

    def curve(self, layer_acts: dict[int, np.ndarray], final: ArrayLike) -> list[dict]:
        """Per-layer decodability + the tuned lens's improvement over the raw logit lens."""
        Yf = np.asarray(final, dtype=np.float64)
        rows = []
        for layer in sorted(layer_acts):
            H = layer_acts[layer]
            rows.append({
                "layer": int(layer),
                "decodability": self.decodability(layer, H, Yf),
                "tuned_residual": self.residual(layer, H, Yf),
                "raw_residual": self.raw_residual(H, Yf),
            })
        return rows
