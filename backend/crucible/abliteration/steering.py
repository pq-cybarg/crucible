from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def steer(activations: ArrayLike, vector: ArrayLike, coefficient: float) -> np.ndarray:
    a = np.asarray(activations, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    return a + coefficient * v
