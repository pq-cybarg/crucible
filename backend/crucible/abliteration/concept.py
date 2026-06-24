from __future__ import annotations
# Concept steering (Representation Engineering / Contrastive Activation Addition).
# Generalizes the refusal-direction machinery to ANY concept: give paired examples that
# do / don't express it (honest vs evasive, verbose vs terse, cheerful vs grim), and the
# steering vector is the difference of their mean activations. Add +c*v to the residual to
# induce the concept, -c*v to suppress it. Refusal abliteration is the special case where
# the concept is "refuse".
import numpy as np
from numpy.typing import ArrayLike


def concept_vector(positive: ArrayLike, negative: ArrayLike, normalize: bool = False) -> np.ndarray:
    """CAA steering vector: mean(positive) - mean(negative). With normalize=True returns a
    unit direction (use the model's own activation norm to scale at injection time)."""
    p = np.asarray(positive, dtype=np.float64)
    n = np.asarray(negative, dtype=np.float64)
    if p.ndim == 1:
        p = p[None, :]
    if n.ndim == 1:
        n = n[None, :]
    if p.shape[0] == 0 or n.shape[0] == 0:
        raise ValueError("need at least one example on each side")
    v = p.mean(axis=0) - n.mean(axis=0)
    if normalize:
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            raise ValueError("concept vector is zero (positive and negative means coincide)")
        return v / norm
    return v


def project_strength(activation: ArrayLike, vector: ArrayLike) -> float:
    """Signed projection of an activation onto the concept vector, normalized by the
    vector's length — how strongly the activation already expresses the concept."""
    v = np.asarray(vector, dtype=np.float64)
    a = np.asarray(activation, dtype=np.float64)
    nv = float(np.linalg.norm(v)) or 1.0
    return float(a @ v / nv)


def separability(positive: ArrayLike, negative: ArrayLike, vector: ArrayLike) -> float:
    """Cohen's-d-style separation of the two sides along the concept vector — a quick read
    on whether the concept is actually linearly encoded (high) or not (~0)."""
    v = np.asarray(vector, dtype=np.float64)
    nv = float(np.linalg.norm(v)) or 1.0
    u = v / nv
    pp = np.asarray(positive, dtype=np.float64) @ u
    nn = np.asarray(negative, dtype=np.float64) @ u
    pooled = float(np.sqrt((pp.var() + nn.var()) / 2.0)) + 1e-9
    return float((pp.mean() - nn.mean()) / pooled)
