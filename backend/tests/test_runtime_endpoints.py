from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Model, Registry


def mkapp(tmp_path):
    reg = Registry(tmp_path / "r.json")
    reg.register(Model(id="gguf-only", name="g", base_id=None, path="/m/missing.gguf",
                       quant="Q4_K_M", kind="base", endpoint=None, created="2026-06-24", notes=""))
    return TestClient(create_app(registry=reg, agent_root=tmp_path))


def test_runtime_status_starts_empty(tmp_path):
    c = mkapp(tmp_path)
    st = c.get("/api/runtime").json()
    assert st["resident"] == [] and st["active"] == []
    assert st["max_resident"] >= 1


def test_start_unknown_model_404(tmp_path):
    c = mkapp(tmp_path)
    assert c.post("/api/runtime/start", json={"model_id": "ghost"}).status_code == 404


def test_start_rejects_missing_gguf_409(tmp_path):
    c = mkapp(tmp_path)
    # path ends in .gguf but the file doesn't exist -> can't launch
    assert c.post("/api/runtime/start", json={"model_id": "gguf-only"}).status_code == 409


def test_set_active_roundtrips(tmp_path):
    c = mkapp(tmp_path)
    st = c.post("/api/runtime/active", json={"model_ids": ["a", "b"]}).json()
    assert st["active"] == ["a", "b"]


def test_stop_absent_is_false(tmp_path):
    c = mkapp(tmp_path)
    assert c.post("/api/runtime/stop", json={"model_id": "nope"}).json()["stopped"] is False
