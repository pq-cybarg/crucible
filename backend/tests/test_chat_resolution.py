"""The forge must resolve a chat model so 'chat with Crucible local' works without an
explicit model= wired into create_app (the run.sh default)."""
from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Model, Registry


def _reg(tmp_path, **endpoint):
    reg = Registry(tmp_path / "r.json")
    reg.register(Model(id="local-gguf", name="local", base_id=None, path="/m/x.gguf",
                       quant="Q4_K_M", kind="base", created="2026-06-24", notes="",
                       endpoint=endpoint.get("endpoint")))
    return reg


class AdapterStub:
    num_layers = 2
    def generate_chat(self, messages, n=256):
        return "adapter says hi"


def test_503_when_nothing_available(tmp_path):
    reg = Registry(tmp_path / "r.json")
    c = TestClient(create_app(registry=reg))
    r = c.post("/api/agent/run", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    assert "no model available" in r.json()["detail"]


def test_falls_back_to_local_adapter(tmp_path):
    reg = Registry(tmp_path / "r.json")
    c = TestClient(create_app(registry=reg, abliteration_adapter=AdapterStub()))
    r = c.post("/api/agent/run", json={"messages": [{"role": "user", "content": "hi"}],
                                       "permissions": {"default": "allow", "modes": {}}})
    assert r.status_code == 200
    assert "adapter says hi" in r.text


def test_model_id_unknown_is_404(tmp_path):
    c = TestClient(create_app(registry=_reg(tmp_path)))
    r = c.post("/api/agent/run", json={"messages": [{"role": "user", "content": "hi"}],
                                       "model_id": "ghost"})
    assert r.status_code == 404


def test_model_id_local_without_adapter_is_409(tmp_path):
    c = TestClient(create_app(registry=_reg(tmp_path)))   # local-gguf has no endpoint, no adapter
    r = c.post("/api/agent/run", json={"messages": [{"role": "user", "content": "hi"}],
                                       "model_id": "local-gguf"})
    assert r.status_code == 409
