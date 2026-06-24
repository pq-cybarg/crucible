from __future__ import annotations
# Activation patching / causal tracing (ROME/MEMIT-style). Where the projection-based
# diagnosis is *correlational* ("the refusal direction is large here"), patching is
# *causal*: run the corrupt (harmful) prompt but splice in the clean (harmless) prompt's
# residual at one layer, and measure how much of the clean->corrupt metric gap that single
# intervention restores. The layer with the largest restoration is the one that *causes*
# the behavior, not merely correlates with it.
import numpy as np
from numpy.typing import ArrayLike


def normalized_restoration(clean: float, corrupt: float, patched: float) -> float:
    """Fraction of the clean->corrupt metric gap recovered by a patch.
    0.0 = no causal effect (patched looks like corrupt); 1.0 = fully causal (restores clean).
    Works regardless of which of clean/corrupt is larger."""
    denom = clean - corrupt
    if denom == 0.0:
        return 0.0
    return float((patched - corrupt) / denom)


def trace_summary(per_layer: list[dict]) -> dict:
    """Identify the most causally responsible site (largest |restoration|)."""
    if not per_layer:
        return {"peak_layer": None, "peak_restoration": 0.0, "per_layer": []}
    peak = max(per_layer, key=lambda d: abs(d["restoration"]))
    return {"peak_layer": int(peak["layer"]),
            "peak_restoration": float(peak["restoration"]),
            "per_layer": per_layer}


def causal_trace(adapter, clean_prompt: str, corrupt_prompt: str,
                 layers: list[int], direction: ArrayLike) -> dict:
    """Per-layer causal trace of the refusal direction via activation patching.
    Metric = projection of the last-token final-residual onto `direction`. Requires a
    torch adapter exposing residual_projection / patched_residual_projection. The pure
    restoration math (normalized_restoration, trace_summary) is unit-tested; this
    orchestration is a model path."""
    r = np.asarray(direction, dtype=np.float64)
    clean = float(adapter.residual_projection(clean_prompt, r))
    corrupt = float(adapter.residual_projection(corrupt_prompt, r))
    per_layer: list[dict] = []
    for layer in layers:
        patched = float(adapter.patched_residual_projection(corrupt_prompt, clean_prompt, layer, r))
        per_layer.append({
            "layer": int(layer),
            "restoration": normalized_restoration(clean, corrupt, patched),
            "clean": clean, "corrupt": corrupt, "patched": patched,
        })
    summary = trace_summary(per_layer)
    summary["clean"] = clean
    summary["corrupt"] = corrupt
    return summary
