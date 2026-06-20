import numpy as np
from numpy.typing import ArrayLike


def compute_refusal_direction(harmful: ArrayLike, harmless: ArrayLike) -> np.ndarray:
    h = np.asarray(harmful, dtype=np.float64)
    l = np.asarray(harmless, dtype=np.float64)
    diff = h.mean(axis=0) - l.mean(axis=0)
    norm = float(np.linalg.norm(diff))
    if norm == 0.0:
        raise ValueError("refusal direction is zero (harmful and harmless means coincide)")
    return diff / norm
