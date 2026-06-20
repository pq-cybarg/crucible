import numpy as np

from crucible.abliteration.diagnosis import (
    ablation_impact, best_layer, explain_mechanism, layer_refusal_profile)


class Adapter:
    hidden_size = 6

    def __init__(self, refusal_layer):
        self.refusal_layer = refusal_layer
        self.e = np.zeros(6)
        self.e[1] = 1.0

    def activations(self, prompts, layer):
        rng = np.random.default_rng(layer)
        out = rng.standard_normal((len(prompts), 6)) * 0.05
        if layer == self.refusal_layer:
            for i, p in enumerate(prompts):
                if "harm" in p:
                    out[i] = out[i] + 6.0 * self.e
        return out


def test_profile_localizes_refusal_layer():
    a = Adapter(refusal_layer=2)
    prof = layer_refusal_profile(a, ["harm0", "harm1"], ["safe0", "safe1"], [0, 1, 2, 3])
    assert best_layer(prof) == 2


def test_ablation_impact_fraction_between_0_and_1():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((6, 9))
    r = np.zeros(6)
    r[1] = 1.0
    imp = ablation_impact(W, r)
    assert 0.0 < imp["removed_fraction"] < 1.0


def test_explain_marks_small_removal_surgical():
    rng = np.random.default_rng(0)
    r = np.zeros(6)
    r[1] = 1.0
    impacts = {"o_proj": ablation_impact(rng.standard_normal((6, 6)), r)}
    prof = [{"layer": 2, "separation": 5.0, "margin": 12.0}]
    report = explain_mechanism(prof, impacts, "glm")
    assert report["best_layer"] == 2
    assert "rank-1" in report["removal"]
    assert isinstance(report["surgical"], bool)
