import numpy as np
from numpy.typing import ArrayLike


def layer_refusal_profile(adapter, harmful: list[str], harmless: list[str],
                          layers: list[int]) -> list[dict]:
    profile: list[dict] = []
    for layer in layers:
        h = np.asarray(adapter.activations(harmful, layer), dtype=np.float64)
        l = np.asarray(adapter.activations(harmless, layer), dtype=np.float64)
        diff = h.mean(axis=0) - l.mean(axis=0)
        separation = float(np.linalg.norm(diff))
        r = diff / (separation or 1.0)
        hp, lp = h @ r, l @ r
        pooled = float(np.sqrt((hp.var() + lp.var()) / 2.0)) + 1e-9
        margin = float((hp.mean() - lp.mean()) / pooled)
        profile.append({"layer": layer, "separation": separation, "margin": margin})
    return profile


def best_layer(profile: list[dict]) -> int:
    return int(max(profile, key=lambda p: p["margin"])["layer"])


def ablation_impact(W: ArrayLike, direction: ArrayLike) -> dict:
    w = np.asarray(W, dtype=np.float64)
    r = np.asarray(direction, dtype=np.float64)
    removed = np.outer(r, r @ w)
    total = float(np.linalg.norm(w))
    removed_norm = float(np.linalg.norm(removed))
    return {"total_norm": total, "removed_norm": removed_norm,
            "removed_fraction": (removed_norm / total) if total else 0.0}


def explain_mechanism(profile: list[dict], matrices_impact: dict, base_id: str) -> dict:
    bl = best_layer(profile) if profile else 0
    fractions = [m["removed_fraction"] for m in matrices_impact.values()]
    mean_removed = sum(fractions) / len(fractions) if fractions else 0.0
    heaviest = (max(matrices_impact.items(), key=lambda kv: kv[1]["removed_fraction"])[0]
                if matrices_impact else None)
    surgical = mean_removed < 0.05
    return {
        "base_id": base_id,
        "best_layer": bl,
        "layer_profile": profile,
        "components": matrices_impact,
        "heaviest_component": heaviest,
        "mean_removed_fraction": mean_removed,
        "surgical": surgical,
        "collateral_risk": "low" if surgical else
            "elevated - refusal is entangled with a large weight component",
        "why": ("Alignment/safety fine-tuning (RLHF + safety SFT) installed a roughly linear "
                "'refusal feature' in the residual stream. When a prompt activates it, the model "
                "is steered toward refusal phrasing."),
        "how": (f"Harmful vs harmless prompts are most linearly separable at layer {bl} "
                "(highest margin). Residual-writing matrices (o_proj, down_proj) add a component "
                "along the refusal direction r; later layers read that component and emit refusal tokens."),
        "removal": ("Orthogonalization subtracts only the rank-1 projection onto r (W - r r^T W). "
                    "The matrix's action on the (d-1)-dimensional subspace orthogonal to r is "
                    "unchanged, so capabilities encoded in other directions are preserved exactly - "
                    "that is why the cut is surgical."),
    }
