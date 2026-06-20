import numpy as np

from crucible.abliteration.sweep import recommend_strength, strength_sweep


def test_recommend_picks_best_net():
    curve = [
        {"strength": 0.0, "harmful_compliance": 0.0, "benign_over_refusal": 0.0},
        {"strength": 0.5, "harmful_compliance": 0.9, "benign_over_refusal": 0.1},  # net 0.8
        {"strength": 1.0, "harmful_compliance": 1.0, "benign_over_refusal": 0.6},  # net 0.4
    ]
    assert recommend_strength(curve) == 0.5


class StubAdapter:
    hidden_size = 4

    def __init__(self):
        self.e = np.array([0.0, 1.0, 0.0, 0.0])
        rng = np.random.default_rng(0)
        self._W = {"o": rng.standard_normal((4, 4)), "d": rng.standard_normal((4, 4))}
        self._base = {k: v.copy() for k, v in self._W.items()}

    def activations(self, prompts, layer):
        out = np.random.default_rng(1).standard_normal((len(prompts), 4)) * 0.05
        for i, p in enumerate(prompts):
            if "harm" in p:
                out[i] = out[i] + 3.0 * self.e
        return out

    def writing_matrices(self):
        return list(self._W)

    def get_matrix(self, n):
        return self._W[n]

    def set_matrix(self, n, W):
        self._W[n] = np.asarray(W)

    def generate(self, p, max_new_tokens=40):
        return "I'm sorry I can't" if "harm" in p else "sure"


def test_sweep_structure_and_restoration():
    a = StubAdapter()
    out = strength_sweep(a, ["harm1", "harm2"], ["ok1", "ok2"], layer=0, strengths=[0.0, 0.5, 1.0])
    assert len(out["curve"]) == 3
    assert out["curve"][0]["strength"] == 0.0
    assert "recommended_strength" in out
    # matrices restored to base after the sweep
    for n in a.writing_matrices():
        assert np.allclose(a.get_matrix(n), a._base[n])
