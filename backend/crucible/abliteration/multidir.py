from __future__ import annotations
# Multiple refusal directions. A single difference-of-means direction (and even a rank-k
# SVD of one activation cloud) assumes refusal lives on one axis. Replications show it
# doesn't: different harm categories trigger refusal along partially-independent directions,
# so a rank-1 cut leaves "sticky" residual refusal. The difference-of-means is, by
# construction, a SINGLE vector — deflating it removes all mean separation — so the honest
# way to recover several directions is the SVD of the per-example difference vectors
# (harmful_i - harmless_i). The dominant singular vector is the global refusal axis;
# later ones capture systematic, category-specific refusal that a single cut misses.
import numpy as np
from numpy.typing import ArrayLike


def refusal_directions(harmful: ArrayLike, harmless: ArrayLike,
                       k: int = 3, min_sv: float = 1e-6) -> tuple[np.ndarray, list[float]]:
    """Up to k orthonormal refusal directions and the singular value (separation strength)
    each carries, from the uncentered SVD of the paired difference vectors. harmful/harmless
    are aligned by index (truncated to the shorter)."""
    h = np.asarray(harmful, dtype=np.float64)
    l = np.asarray(harmless, dtype=np.float64)
    if h.ndim == 1:
        h = h[None, :]
    if l.ndim == 1:
        l = l[None, :]
    n = min(h.shape[0], l.shape[0])
    dim = h.shape[-1]
    if n == 0:
        return np.zeros((0, dim)), []
    diff = h[:n] - l[:n]                       # per-example difference vectors
    _, s, vt = np.linalg.svd(diff, full_matrices=False)
    keep = [i for i in range(len(s)) if float(s[i]) > min_sv][: max(0, k)]
    if not keep:
        return np.zeros((0, dim)), []
    return vt[keep], [float(s[i]) for i in keep]


def sticky_fraction(seps: list[float]) -> float:
    """Share of refusal separation living BEYOND the primary axis (0 = perfectly rank-1).
    High -> a single-direction abliteration will under-remove refusal."""
    if not seps:
        return 0.0
    total = sum(s * s for s in seps)
    if total == 0.0:
        return 0.0
    return float(1.0 - (seps[0] * seps[0]) / total)
