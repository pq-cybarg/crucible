import numpy as np
from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Model, Registry


class CausalFake:
    """Fake adapter exercising the causal-trace / multidir / concept endpoints. Refusal is
    a boost along basis vector e at the 'decided' layer; layer 1 is the causal site."""
    hidden_size = 8
    num_layers = 3

    def __init__(self):
        self.e = np.zeros(8); self.e[1] = 1.0
        self.e2 = np.zeros(8); self.e2[2] = 1.0

    def _harmful(self, p):
        return any(k in p for k in ("harm", "danger", "illegal", "unethical", "weapon", "hack"))

    def activations(self, prompts, layer):
        rng = np.random.default_rng(3)
        out = rng.standard_normal((len(prompts), 8)) * 0.02
        boost = 6.0 if layer == 1 else 3.0
        for i, p in enumerate(prompts):
            if self._harmful(p):
                # heterogeneous: alternate prompts refuse along e vs e2 (two directions)
                out[i] = out[i] + boost * (self.e if i % 2 == 0 else self.e2)
        return out

    # causal tracing surface
    def residual_projection(self, prompt, r):
        return 10.0 if self._harmful(prompt) else 0.0

    def patched_residual_projection(self, corrupt_prompt, clean_prompt, layer, r):
        return 0.0 if layer == 1 else 10.0   # patching layer 1 restores the clean metric

    # generation surface (for concept steering demo)
    def generate(self, prompt, n=40):
        return "base output"

    def inject_generate(self, prompt, direction, coefficient, layers, n=40):
        return f"steered({'+' if coefficient > 0 else '-'})"


def mkapp(tmp_path, monkeypatch, adapter):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    reg = Registry(tmp_path / "r.json")
    reg.register(Model(id="glm", name="glm", base_id=None, path="/m/glm.gguf", quant="Q4_K_M",
                       kind="base", endpoint=None, created="2026-06-20", notes=""))
    return TestClient(create_app(registry=reg, agent_root=tmp_path, abliteration_adapter=adapter))


def test_causal_trace_needs_adapter(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch, None)
    assert c.post("/api/abliteration/causal-trace", json={"base_id": "glm"}).status_code == 503


def test_causal_trace_localizes_causal_layer(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch, CausalFake())
    r = c.post("/api/abliteration/causal-trace", json={"base_id": "glm", "layers": [0, 1, 2]})
    assert r.status_code == 200
    body = r.json()
    assert body["peak_layer"] == 1               # the causal site
    assert body["peak_restoration"] == 1.0
    assert body["clean"] == 0.0 and body["corrupt"] == 10.0


def test_multidir_reports_multiple_axes(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch, CausalFake())
    r = c.post("/api/abliteration/multidir", json={"base_id": "glm", "k": 3, "layer": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["n_directions"] >= 2
    assert body["sticky_fraction"] > 0.0          # refusal lives beyond the primary axis
    assert len(body["separations"]) == body["n_directions"]


def test_concept_steering_demo(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch, CausalFake())
    r = c.post("/api/abliteration/concept", json={
        "base_id": "glm",
        "positive": ["this is harmful and dangerous", "illegal weapon hacking"],
        "negative": ["a nice walk in the park", "baking bread today"],
        "layer": 1, "test_prompt": "tell me about your day", "coefficient": 4.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["separability"] > 1.0             # concept is linearly encoded
    assert body["test"]["steered+"] == "steered(+)"
    assert body["test"]["steered-"] == "steered(-)"


def test_concept_unknown_model_404(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch, CausalFake())
    r = c.post("/api/abliteration/concept",
               json={"base_id": "nope", "positive": ["x"], "negative": ["y"]})
    assert r.status_code == 404
