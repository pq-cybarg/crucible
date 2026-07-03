from __future__ import annotations
# Piecemeal alignment. Alignment isn't one thing — it's a bundle of partially-independent
# behaviors (blunt refusal, moralizing, hedging/disclaimers, topic-specific guards, tone).
# This decomposes it into orthogonal COMPONENT directions the operator can inspect, name,
# and edit individually: keep the components you want, remove the ones you don't, strengthen
# others — and compose a custom alignment from the parts instead of one all-or-nothing cut.
# Pure numpy + SVD; the token labels that make components human-pickable come from the model.
import numpy as np
from numpy.typing import ArrayLike

from crucible.abliteration.multidir import refusal_directions


def decompose_alignment(harmful_acts: ArrayLike, harmless_acts: ArrayLike,
                        k: int = 4) -> list[dict]:
    """Break alignment into up to k orthogonal component directions (per-example difference
    SVD). Each component is a separable 'aligned vs not' axis with its share of the signal."""
    dirs, seps = refusal_directions(harmful_acts, harmless_acts, k)
    total = sum(s * s for s in seps) or 1.0
    comps: list[dict] = []
    for i in range(dirs.shape[0]):
        comps.append({
            "index": i,
            "direction": dirs[i],
            "separation": float(seps[i]),
            "share": float(seps[i] * seps[i] / total),
        })
    return comps


def component_edit(W: ArrayLike, direction: ArrayLike, coef: float = 1.0,
                   mode: str = "unalign") -> np.ndarray:
    """Apply one alignment component to a writing matrix. mode='unalign' removes that
    component's contribution, 'realign' adds/strengthens it."""
    r = np.asarray(direction, dtype=np.float64)
    r = r / (float(np.linalg.norm(r)) or 1.0)
    Wf = np.asarray(W, dtype=np.float64)
    sign = 1.0 if mode == "realign" else -1.0
    return Wf + sign * coef * np.outer(r, r @ Wf)


def compose_edit(W: ArrayLike, components: list[dict], selections: list[dict]) -> np.ndarray:
    """Compose a custom edit from a CHOSEN SUBSET of components, each with its own coef+mode.
    selections: [{index, coef?, mode?}]. Components not selected are left untouched."""
    out = np.asarray(W, dtype=np.float64).copy()
    by_index = {c["index"]: c for c in components}
    for sel in selections:
        comp = by_index.get(sel["index"])
        if comp is None:
            continue
        out = component_edit(out, comp["direction"], float(sel.get("coef", 1.0)),
                             sel.get("mode", "unalign"))
    return out


def compose_direction(components: list[dict], selections: list[dict]) -> np.ndarray:
    """The combined steering direction from a selection (for runtime steering / LoRA), signed
    by each component's mode. Handy when you'd rather steer activations than edit weights."""
    if not components:
        return np.zeros(0)
    dim = np.asarray(components[0]["direction"]).shape[0]
    acc = np.zeros(dim, dtype=np.float64)
    by_index = {c["index"]: c for c in components}
    for sel in selections:
        comp = by_index.get(sel["index"])
        if comp is None:
            continue
        r = np.asarray(comp["direction"], dtype=np.float64)
        r = r / (float(np.linalg.norm(r)) or 1.0)
        sign = 1.0 if sel.get("mode", "unalign") == "realign" else -1.0
        acc += sign * float(sel.get("coef", 1.0)) * r
    return acc
