import numpy as np

from crucible.abliteration.patching import (
    causal_trace, normalized_restoration, trace_summary)


def test_restoration_bounds():
    assert normalized_restoration(10.0, 0.0, 0.0) == 0.0    # patched == corrupt -> no effect
    assert normalized_restoration(10.0, 0.0, 10.0) == 1.0   # patched == clean -> fully causal
    assert normalized_restoration(10.0, 0.0, 5.0) == 0.5
    assert normalized_restoration(0.0, 5.0, 2.5) == 0.5     # works when corrupt > clean
    assert normalized_restoration(3.0, 3.0, 3.0) == 0.0     # zero gap -> defined as 0


def test_trace_summary_picks_largest_magnitude():
    per_layer = [
        {"layer": 0, "restoration": 0.1},
        {"layer": 5, "restoration": -0.9},   # biggest magnitude (sign-agnostic)
        {"layer": 9, "restoration": 0.4},
    ]
    s = trace_summary(per_layer)
    assert s["peak_layer"] == 5
    assert s["peak_restoration"] == -0.9


def test_trace_summary_empty():
    assert trace_summary([])["peak_layer"] is None


class _FakeAdapter:
    """Toy adapter: layer L is the causal site — patching it restores the clean metric;
    every other layer leaves the corrupt metric unchanged."""
    def __init__(self, clean=10.0, corrupt=0.0, causal_layer=4):
        self._clean, self._corrupt, self._causal = clean, corrupt, causal_layer

    def residual_projection(self, prompt, r):
        return self._clean if prompt == "harmless" else self._corrupt

    def patched_residual_projection(self, corrupt_prompt, clean_prompt, layer, r):
        return self._clean if layer == self._causal else self._corrupt


def test_causal_trace_localizes_the_causal_layer():
    out = causal_trace(_FakeAdapter(causal_layer=4), "harmless", "harmful",
                       layers=[0, 2, 4, 6], direction=np.array([1.0, 0.0]))
    assert out["peak_layer"] == 4
    assert out["peak_restoration"] == 1.0
    assert {d["layer"]: round(d["restoration"], 3) for d in out["per_layer"]} == {
        0: 0.0, 2: 0.0, 4: 1.0, 6: 0.0}
