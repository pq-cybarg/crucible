# Rank-k refusal subspace. r1 = normalized difference-of-means (the primary refusal
# axis); r2..rk = PCA of the harmful activations orthogonal to r1 (the "sticky"
# residual a rank-1 cut misses). Returns orthonormal directions + explained variance
# (the defensibility metric). Deterministic -> reproducible.
import numpy as np
from numpy.typing import ArrayLike


def refusal_subspace(harmful: ArrayLike, harmless: ArrayLike, k: int = 1) -> tuple[np.ndarray, list[float]]:
    h = np.asarray(harmful, dtype=np.float64)
    l = np.asarray(harmless, dtype=np.float64)
    diff = h.mean(axis=0) - l.mean(axis=0)
    norm = float(np.linalg.norm(diff))
    if norm == 0.0:
        raise ValueError("primary refusal direction is zero")
    r1 = diff / norm
    if k <= 1:
        return r1[None, :], [1.0]

    centered = h - h.mean(axis=0)
    centered = centered - np.outer(centered @ r1, r1)  # remove r1 component
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    extra = vt[: k - 1]
    dirs = np.vstack([r1, extra])
    total = float((s ** 2).sum()) or 1.0
    explained = [1.0] + [float((s[i] ** 2) / total) for i in range(min(k - 1, len(s)))]
    return dirs, explained
