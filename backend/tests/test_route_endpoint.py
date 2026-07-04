from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Model, Registry


def mkapp(tmp_path):
    reg = Registry(tmp_path / "r.json")
    reg.register(Model(id="qwen-coder", name="qwen2.5-coder", base_id=None, path="/m/c.gguf",
                       quant="Q4_K_M", kind="base", endpoint="http://x", created="2026-07-04", notes=""))
    reg.register(Model(id="mini", name="qwen2.5-0.5b", base_id=None, path="/m/m.gguf",
                       quant="Q4_K_M", kind="base", endpoint="http://y", created="2026-07-04", notes=""))
    return TestClient(create_app(registry=reg, agent_root=tmp_path))


def test_route_code_to_coder(tmp_path, monkeypatch):
    # both endpoints "unreachable" -> avail False; but classification + candidates still returned
    c = mkapp(tmp_path)
    r = c.post("/api/route", json={"prompt": "refactor this python function", "user_level": "max"}).json()
    assert r["task"] == "code"
    assert any(m["id"] == "qwen-coder" and "code" in m["tags"] for m in r["candidates"])


def test_route_chat_default(tmp_path):
    r = mkapp(tmp_path).post("/api/route", json={"prompt": "hey there"}).json()
    assert r["task"] == "chat"
