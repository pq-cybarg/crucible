import numpy as np

from crucible.abliteration.components import (
    compose_direction, compose_edit, component_edit, decompose_alignment)


def _hetero(n=200, seed=0):
    """Two independent alignment components: half the harmful prompts separate on axis 0,
    half on axis 1 — so decomposition should surface two distinct components."""
    rng = np.random.default_rng(seed)
    dim = 6
    harmless = rng.normal(0, 0.02, (n, dim))
    harmful = harmless.copy()
    half = n // 2
    harmful[:half, 0] += 1.0
    harmful[half:, 1] += 0.7
    return harmful, harmless


def test_decompose_finds_multiple_components():
    comps = decompose_alignment(*_hetero(), k=3)
    assert len(comps) >= 2
    axes = {int(np.argmax(np.abs(comps[0]["direction"]))),
            int(np.argmax(np.abs(comps[1]["direction"])))}
    assert axes == {0, 1}
    assert comps[0]["share"] >= comps[1]["share"]     # ranked by contribution
    assert abs(sum(c["share"] for c in comps) - 1.0) < 1e-6


def test_component_edit_modes():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((8, 5)); r = rng.standard_normal(8); r /= np.linalg.norm(r)
    assert np.allclose(r @ component_edit(W, r, 1.0, "unalign"), 0.0, atol=1e-8)
    assert np.linalg.norm(r @ component_edit(W, r, 1.0, "realign")) > np.linalg.norm(r @ W)


def test_compose_applies_only_selected_components():
    # two orthogonal components; removing only #0 clears axis-0 refusal, leaves axis-1 intact
    e0 = np.zeros(6); e0[0] = 1.0
    e1 = np.zeros(6); e1[1] = 1.0
    comps = [{"index": 0, "direction": e0}, {"index": 1, "direction": e1}]
    rng = np.random.default_rng(2)
    W = rng.standard_normal((6, 4))
    out = compose_edit(W, comps, [{"index": 0, "coef": 1.0, "mode": "unalign"}])
    assert np.allclose(e0 @ out, 0.0, atol=1e-8)          # component 0 removed
    assert np.allclose(e1 @ out, e1 @ W, atol=1e-8)       # component 1 untouched


def test_compose_multiple_selections():
    e0 = np.zeros(4); e0[0] = 1.0
    e1 = np.zeros(4); e1[1] = 1.0
    comps = [{"index": 0, "direction": e0}, {"index": 1, "direction": e1}]
    W = np.random.default_rng(3).standard_normal((4, 4))
    out = compose_edit(W, comps, [{"index": 0, "mode": "unalign"}, {"index": 1, "mode": "unalign"}])
    assert np.allclose(e0 @ out, 0.0, atol=1e-8)
    assert np.allclose(e1 @ out, 0.0, atol=1e-8)


def test_compose_direction_signs():
    e0 = np.zeros(3); e0[0] = 1.0
    e1 = np.zeros(3); e1[1] = 1.0
    comps = [{"index": 0, "direction": e0}, {"index": 1, "direction": e1}]
    d = compose_direction(comps, [{"index": 0, "mode": "unalign", "coef": 2.0},
                                  {"index": 1, "mode": "realign", "coef": 1.0}])
    assert d[0] == -2.0 and d[1] == 1.0                   # unalign negative, realign positive
