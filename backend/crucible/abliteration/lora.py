from __future__ import annotations
# LoRA un-alignment. Abliteration edits base weights in place; a LoRA instead carries the
# change as a small, PORTABLE low-rank adapter (ΔW = (alpha/rank)·B·A) you can attach or
# detach at inference without touching the base model — the peft-style complement to a
# permanent cut. Because the refusal-removal update is itself low-rank (rank-1 for a single
# direction), a LoRA can represent it exactly; higher rank captures multi-direction removal
# or a data-trained un-alignment. Pure numpy + SVD, so it's deterministic and unit-tested.
import numpy as np
from numpy.typing import ArrayLike


class LoRA:
    """Low-rank weight delta: ΔW = (alpha/rank) · B @ A.  A:(rank,in)  B:(out,rank)."""

    def __init__(self, A: ArrayLike, B: ArrayLike, alpha: float | None = None):
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.rank = int(self.A.shape[0])
        self.alpha = float(alpha) if alpha is not None else float(self.rank)

    def delta(self) -> np.ndarray:
        return (self.alpha / self.rank) * (self.B @ self.A)

    def apply(self, W: ArrayLike) -> np.ndarray:
        return np.asarray(W, dtype=np.float64) + self.delta()

    @property
    def n_params(self) -> int:
        return self.A.size + self.B.size


def fit_lowrank(delta_W: ArrayLike, rank: int, alpha: float | None = None) -> LoRA:
    """Best rank-r factorization of a target ΔW (truncated SVD). The returned LoRA's delta()
    reproduces the rank-r approximation of ΔW regardless of the chosen alpha."""
    D = np.asarray(delta_W, dtype=np.float64)
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    r = int(min(max(1, rank), len(s)))
    al = float(alpha) if alpha is not None else float(r)
    B = U[:, :r] * s[:r]                     # (out, r)  == U·diag(s)
    A = Vt[:r] * (r / al)                     # (r, in)   folds the alpha/rank scaling
    return LoRA(A, B, alpha=al)


def alignment_lora(W: ArrayLike, direction: ArrayLike, coef: float = 1.0, rank: int = 1,
                   alpha: float | None = None, mode: str = "unalign") -> LoRA:
    """A portable adapter that edits the refusal component of one writing matrix, in either
    direction:
      mode='unalign' -> dW = -coef * r (rT W)   (REMOVE refusal; attach = uncensored)
      mode='realign' -> dW = +coef * r (rT W)   (RESTORE/strengthen refusal; re-install safety)
    Detach to restore the original either way."""
    r = np.asarray(direction, dtype=np.float64)
    r = r / (float(np.linalg.norm(r)) or 1.0)
    Wm = np.asarray(W, dtype=np.float64)
    sign = 1.0 if mode == "realign" else -1.0
    dW = sign * coef * np.outer(r, r @ Wm)
    return fit_lowrank(dW, rank, alpha)


def unalign_lora(W: ArrayLike, direction: ArrayLike, coef: float = 1.0,
                 rank: int = 1, alpha: float | None = None) -> LoRA:
    """Un-alignment adapter (REMOVE refusal). See alignment_lora(mode='unalign')."""
    return alignment_lora(W, direction, coef, rank, alpha, mode="unalign")


def realign_lora(W: ArrayLike, direction: ArrayLike, coef: float = 1.0,
                 rank: int = 1, alpha: float | None = None) -> LoRA:
    """Realignment adapter (RESTORE/strengthen refusal). See alignment_lora(mode='realign')."""
    return alignment_lora(W, direction, coef, rank, alpha, mode="realign")


def reconstruction_error(lora: LoRA, target: ArrayLike) -> float:
    """Relative error between the LoRA's delta and a target ΔW (0 = exact)."""
    T = np.asarray(target, dtype=np.float64)
    denom = float(np.linalg.norm(T)) or 1.0
    return float(np.linalg.norm(lora.delta() - T) / denom)
