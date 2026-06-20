import numpy as np
from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Model, Registry


class FakeAdapter:
    hidden_size = 8
    num_layers = 3

    def __init__(self):
        rng = np.random.default_rng(0)
        self.e = np.zeros(8)
        self.e[1] = 1.0
        self._mats = {"o_proj": rng.standard_normal((8, 8))}
        self.saved = None

    def activations(self, prompts, layer):
        rng = np.random.default_rng(3)
        out = rng.standard_normal((len(prompts), 8)) * 0.05
        boost = 6.0 if layer == 1 else 3.0
        for i, p in enumerate(prompts):
            if any(k in p for k in ("harm", "danger", "illegal", "unethical")):
                out[i] = out[i] + boost * self.e
        return out

    def writing_matrices(self):
        return list(self._mats)

    def get_matrix(self, name):
        return self._mats[name]

    def set_matrix(self, name, W):
        self._mats[name] = W

    def save(self, path):
        self.saved = path


def mkapp(tmp_path, monkeypatch, adapter=None):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    reg = Registry(tmp_path / "r.json")
    reg.register(Model(id="glm", name="glm", base_id=None, path="/m/glm.gguf", quant="Q4_K_M",
                       kind="base", endpoint=None, created="2026-06-20", notes=""))
    return create_app(registry=reg, agent_root=tmp_path, abliteration_adapter=adapter)


def test_promptsets(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch))
    body = c.get("/api/abliteration/promptsets").json()
    assert len(body["harmful"]) > 0 and len(body["harmless"]) > 0


def test_diagnose_requires_adapter(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch))
    assert c.post("/api/abliteration/diagnose", json={"base_id": "glm"}).status_code == 503


def test_diagnose_localizes_and_explains(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch, adapter=FakeAdapter()))
    rep = c.post("/api/abliteration/diagnose", json={"base_id": "glm"}).json()
    assert rep["best_layer"] == 1
    assert "rank-1" in rep["removal"]
    assert "o_proj" in rep["components"]
    assert isinstance(rep["surgical"], bool)


def test_run_requires_adapter(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch))
    r = c.post("/api/abliteration/run", json={"base_id": "glm", "variant_id": "glm-abl"})
    assert r.status_code == 503


def test_run_produces_variant(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch, adapter=FakeAdapter()))
    r = c.post("/api/abliteration/run", json={"base_id": "glm", "variant_id": "glm-abl", "layer": 1})
    assert r.status_code == 200
    out = r.json()
    assert out["variant"]["kind"] == "abliterated"
    assert out["card"]["base_id"] == "glm"
    assert [m["id"] for m in c.get("/api/models/glm-abl/lineage").json()] == ["glm", "glm-abl"]


def test_run_unknown_base_404(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch, adapter=FakeAdapter()))
    r = c.post("/api/abliteration/run", json={"base_id": "nope", "variant_id": "x"})
    assert r.status_code == 404
