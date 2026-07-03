from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


class Adapter:
    num_layers = 2
    def generate_chat(self, messages, max_tokens=256, band_dirs=None, coefficient=1.0):
        return "gateway reply"


def mkapp(tmp_path, adapter=None, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"),
                                 agent_root=tmp_path, abliteration_adapter=adapter))


def test_v1_models_lists_adapter(tmp_path):
    data = mkapp(tmp_path, Adapter()).get("/v1/models").json()["data"]
    assert any(m["id"] == "crucible" for m in data)


def test_v1_chat_routes_to_adapter(tmp_path):
    c = mkapp(tmp_path, Adapter())
    r = c.post("/v1/chat/completions", json={"model": "auto",
               "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "crucible"
    assert body["choices"][0]["message"]["content"] == "gateway reply"
    assert "requested" in body["system_fingerprint"] or "nearest" in body["system_fingerprint"]


def test_v1_chat_streaming_shape(tmp_path):
    c = mkapp(tmp_path, Adapter())
    r = c.post("/v1/chat/completions", json={"model": "crucible", "stream": True,
               "messages": [{"role": "user", "content": "hi"}]})
    assert "data:" in r.text and "[DONE]" in r.text
    assert "gateway reply" in r.text


def test_v1_chat_503_when_no_model(tmp_path):
    c = mkapp(tmp_path)      # no adapter, no endpoints
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503


def test_preferences_roundtrip(tmp_path, monkeypatch):
    c = mkapp(tmp_path, Adapter(), monkeypatch)      # isolated data dir
    assert c.get("/api/provider/preferences").json()["preferences"] == []
    r = c.post("/api/provider/preferences", json={"preferences": ["crucible", "x"]})
    assert r.json()["preferences"] == ["crucible", "x"]
    assert c.get("/api/provider/preferences").json()["preferences"] == ["crucible", "x"]


class ReactAdapter:
    """Adapter that has no native tools; emits text ReAct when it sees the tool preamble."""
    num_layers = 2
    def generate_chat(self, messages, max_tokens=256, band_dirs=None, coefficient=1.0):
        sys = messages[0]["content"] if messages else ""
        if "Action:" in sys:   # tool preamble present -> emit a ReAct action
            return 'Thought: list it\nAction: list_dir\nAction Input: {"path": "."}'
        return "plain reply"


def test_adapter_supports_tools_via_react_bridge(tmp_path):
    c = mkapp(tmp_path, ReactAdapter())
    r = c.post("/v1/chat/completions", json={
        "model": "crucible",
        "messages": [{"role": "user", "content": "list files"}],
        "tools": [{"type": "function", "function": {"name": "list_dir",
                   "description": "list a dir", "parameters": {"type": "object",
                   "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    }).json()
    choice = r["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "list_dir"
    assert "." in tc["function"]["arguments"]


def test_adapter_no_tools_returns_content(tmp_path):
    c = mkapp(tmp_path, ReactAdapter())
    r = c.post("/v1/chat/completions", json={"model": "crucible",
               "messages": [{"role": "user", "content": "hi"}]}).json()
    assert r["choices"][0]["message"]["content"] == "plain reply"
    assert r["choices"][0]["finish_reason"] == "stop"
