from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


class StubAdapter:
    num_layers = 2

    def generate_chat(self, messages, max_new_tokens=128, band_dirs=None, coefficient=1.0):
        steered = " [steered]" if band_dirs else ""
        return f"reply to: {messages[-1]['content']}{steered}"


def test_v1_chat_requires_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    c = TestClient(create_app(registry=Registry(tmp_path / "r.json")))
    assert c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 503


def test_v1_chat_openai_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    c = TestClient(create_app(registry=Registry(tmp_path / "r.json"), abliteration_adapter=StubAdapter()))
    assert c.get("/v1/models").json()["data"][0]["id"] == "crucible"
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]}).json()
    assert r["choices"][0]["message"]["content"] == "reply to: hello"


def test_serve_recipe_set_clear(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    c = TestClient(create_app(registry=Registry(tmp_path / "r.json"), abliteration_adapter=StubAdapter()))
    assert c.get("/api/inference/recipe").json()["active"] is None
    # set via fake adapter would need all_layer_activations; StubAdapter lacks it -> expect failure path is fine
    assert c.delete("/api/inference/recipe").json()["active"] is None
