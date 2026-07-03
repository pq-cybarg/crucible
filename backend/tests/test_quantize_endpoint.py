import numpy as np
from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Model, Registry


class QAdapter:
    num_layers = 2
    def __init__(self):
        rng = np.random.default_rng(0)
        self._m = {"o_proj": rng.standard_normal((32, 32)).astype("float32")}
    def writing_matrices(self): return list(self._m)
    def get_matrix(self, n): return self._m[n]


def mkapp(tmp_path, adapter=None):
    reg = Registry(tmp_path / "r.json")
    reg.register(Model(id="m", name="m", base_id=None, path="/x.gguf", quant="Q8_0",
                       kind="base", endpoint=None, created="2026-07-03", notes=""))
    return TestClient(create_app(registry=reg, agent_root=tmp_path, abliteration_adapter=adapter))


def test_quantize_report_supported(tmp_path):
    c = mkapp(tmp_path, QAdapter())
    r = c.post("/api/weights/quantize", json={"base_id": "m", "dtype": "F16"}).json()
    assert r["supported"] is True and r["n_matrices"] == 1
    assert r["mean_fidelity"] > 0.99


def test_quantize_report_kquant_unsupported(tmp_path):
    c = mkapp(tmp_path, QAdapter())
    r = c.post("/api/weights/quantize", json={"base_id": "m", "dtype": "Q4_K"}).json()
    assert r["supported"] is False


def test_quantize_needs_adapter(tmp_path):
    assert mkapp(tmp_path).post("/api/weights/quantize", json={"base_id": "m"}).status_code == 503
