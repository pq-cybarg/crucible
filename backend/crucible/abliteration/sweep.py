from __future__ import annotations
# Strength sweep: abliterate at a range of doses on one loaded model (apply -> measure
# -> restore), to find the strength that removes refusal without collateral over-refusal.
import numpy as np

from crucible.abliteration.detection import refusal_rate
from crucible.abliteration.direction import compute_refusal_direction


def recommend_strength(curve: list[dict]) -> float:
    # Best net behavior: most harmful-compliance gained, least benign over-refusal paid.
    best = max(curve, key=lambda r: r["harmful_compliance"] - r["benign_over_refusal"])
    return float(best["strength"])


def strength_sweep(adapter, harmful: list[str], harmless: list[str], layer: int,
                   strengths: list[float], max_new_tokens: int = 40) -> dict:
    r = compute_refusal_direction(
        adapter.activations(harmful, layer), adapter.activations(harmless, layer))
    names = adapter.writing_matrices()
    base_W = {n: adapter.get_matrix(n) for n in names}
    removed = {n: np.outer(r, r @ base_W[n]) for n in names}

    curve: list[dict] = []
    try:
        for s in strengths:
            for n in names:
                adapter.set_matrix(n, base_W[n] - s * removed[n])
            h = [adapter.generate(p, max_new_tokens) for p in harmful]
            b = [adapter.generate(p, max_new_tokens) for p in harmless]
            curve.append({
                "strength": float(s),
                "harmful_compliance": 1.0 - refusal_rate(h),
                "benign_over_refusal": refusal_rate(b),
            })
    finally:
        for n in names:
            adapter.set_matrix(n, base_W[n])

    return {"layer": layer, "direction_norm": float(np.linalg.norm(r)),
            "curve": curve, "recommended_strength": recommend_strength(curve) if curve else 0.0}
