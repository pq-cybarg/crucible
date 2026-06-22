from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def project_out(x: ArrayLike, direction: ArrayLike) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64)
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        return a - (a @ d) * d
    return a - np.outer(a @ d, d)


def orthogonalize_writing_matrix(W: ArrayLike, direction: ArrayLike) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64)
    w = np.asarray(W, dtype=np.float64)
    return w - np.outer(d, d @ w)


def orthogonalize_embedding(E: ArrayLike, direction: ArrayLike) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64)
    e = np.asarray(E, dtype=np.float64)
    return e - np.outer(e @ d, d)
