import numpy as np

from crucible.abliteration.pipeline import AbliterationPipeline
from crucible.registry import Model, Registry


class FakeAdapter:
    hidden_size = 8

    def __init__(self):
        rng = np.random.default_rng(0)
        self.e = np.zeros(8)
        self.e[2] = 1.0
        self._mats = {"o_proj": rng.standard_normal((8, 8)),
                      "down_proj": rng.standard_normal((8, 16))}
        self.saved = None

    def activations(self, prompts, layer):
        rng = np.random.default_rng(7)
        out = rng.standard_normal((len(prompts), self.hidden_size)) * 0.05
        for i, p in enumerate(prompts):
            if "harm" in p:
                out[i] = out[i] + 5.0 * self.e
        return out

    def writing_matrices(self):
        return list(self._mats)

    def get_matrix(self, name):
        return self._mats[name]

    def set_matrix(self, name, W):
        self._mats[name] = W

    def save(self, path):
        self.saved = path


def base_model(reg):
    m = Model(id="base", name="base", base_id=None, path="/m/base.gguf", quant="Q4_K_M",
              kind="base", endpoint=None, created="2026-06-20", notes="")
    reg.register(m)
    return m


def test_pipeline_finds_and_removes_refusal(tmp_path):
    reg = Registry(tmp_path / "r.json")
    base = base_model(reg)
    adapter = FakeAdapter()
    pipe = AbliterationPipeline(adapter, reg)
    harmful = [f"harm{i}" for i in range(16)]
    harmless = [f"safe{i}" for i in range(16)]
    variant, card, direction = pipe.abliterate(
        base, harmful, harmless, layer=0, out_path="/m/base-abl.gguf",
        variant_id="base-abl", strength=1.0)
    assert abs(np.dot(direction, adapter.e)) > 0.99
    for name in adapter.writing_matrices():
        assert np.allclose(adapter.e @ adapter.get_matrix(name), 0.0, atol=1e-9)
    assert variant.kind == "abliterated"
    assert [m.id for m in reg.lineage("base-abl")] == ["base", "base-abl"]
    assert card["repro_hash"] and adapter.saved == "/m/base-abl.gguf"


def test_original_untouched(tmp_path):
    reg = Registry(tmp_path / "r.json")
    base = base_model(reg)
    pipe = AbliterationPipeline(FakeAdapter(), reg)
    pipe.abliterate(base, ["harm"], ["safe"], 0, "/m/base-abl.gguf", "base-abl")
    assert reg.get("base").path == "/m/base.gguf"
