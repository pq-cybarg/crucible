from crucible.abliteration.tune import autotune, layer_band, recipe_hash


def test_layer_band():
    assert layer_band(24, "late_half") == list(range(12, 24))
    assert layer_band(24, "last_quarter") == list(range(18, 24))
    assert layer_band(4, "all") == [0, 1, 2, 3]


def test_recipe_hash_stable():
    assert recipe_hash({"a": 1, "b": 2}) == recipe_hash({"b": 2, "a": 1})
    assert len(recipe_hash({"x": 1})) == 16


class StubAdapter:
    num_layers = 4
    import numpy as _np

    def all_layer_activations(self, prompts):
        import numpy as np
        e = np.zeros(4); e[1] = 1.0
        rows = []
        for p in prompts:
            layers = []
            for layer in range(self.num_layers + 1):
                v = np.random.default_rng(layer).standard_normal(4) * 0.05
                if "harm" in p:
                    v = v + 3.0 * e
                layers.append(v)
            rows.append(layers)
        return np.array(rows)

    def generate(self, p, n=22):
        return "I'm sorry I can't" if "harm" in p else "sure"

    def ablate_generate_banded(self, p, band_dirs, coefficient=1.0, n=22):
        # pretend ablation fully unhinges when the band is non-empty
        return "here you go" if "harm" in p else "sure"


def test_autotune_picks_best_and_hashes():
    a = StubAdapter()
    cfgs = [{"band": "all", "rank": 1, "coefficient": 1.0},
            {"band": "late_half", "rank": 2, "coefficient": 1.0}]
    out = autotune(a, ["harm1", "harm2"], ["ok1", "ok2"], cfgs, max_new_tokens=4)
    assert out["baseline"]["harmful_refusal"] == 1.0
    assert out["best"]["harmful_refusal"] == 0.0
    assert len(out["recipe_hash"]) == 16 and out["weights_modified"] is False
