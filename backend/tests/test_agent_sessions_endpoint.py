"""Agent-session endpoints: tabs (dirs/subagents) + loadable memory/context slots + live context."""
from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


def mkapp(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"), agent_root=tmp_path))


def test_tabs_subagents_and_slots_flow(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch)
    # open two agent tabs in different directories
    a = c.post("/api/agent-sessions", json={"title": "web", "cwd": "/proj/web"}).json()
    b = c.post("/api/agent-sessions", json={"title": "api", "cwd": "/proj/api", "model_id": "qwen"}).json()
    assert a["cwd"] == "/proj/web" and b["model_id"] == "qwen"
    # a subagent under the first tab
    sub = c.post("/api/agent-sessions", json={"title": "helper", "cwd": "/proj/web", "parent_id": a["id"]}).json()
    assert sub["parent_id"] == a["id"]
    # top-level view hides subagents; the parent's children show it
    top = [s["id"] for s in c.get("/api/agent-sessions", params={"top": True}).json()["sessions"]]
    assert a["id"] in top and b["id"] in top and sub["id"] not in top
    kids = [s["id"] for s in c.get("/api/agent-sessions", params={"parent": a["id"]}).json()["sessions"]]
    assert kids == [sub["id"]]

    # give the subagent some conversation, then LOAD it as context into tab b + load a memory
    c.patch(f"/api/agent-sessions/{sub['id']}", json={"messages": [{"role": "user", "content": "sub findings"}]})
    c.post(f"/api/agent-sessions/{b['id']}/slots", json={"kind": "context", "ref": sub["id"], "label": "helper"})
    # a crystallized memory to load
    from crucible.memory import MemoryStore
    from crucible.config import get_settings
    MemoryStore(get_settings().data_dir / "memory").crystallize(
        [{"role": "user", "content": "x"}], "abliteration removes refusal", label="abl")
    c.post(f"/api/agent-sessions/{b['id']}/slots", json={"kind": "memory", "ref": "m-0001"})
    c.patch(f"/api/agent-sessions/{b['id']}", json={"messages": [{"role": "user", "content": "do the task"}]})

    # live context injects both loaded slots ahead of the conversation
    ctx = "\n".join(m["content"] for m in c.get(f"/api/agent-sessions/{b['id']}/context").json()["messages"])
    assert "sub findings" in ctx and "abliteration removes refusal" in ctx and ctx.rstrip().endswith("do the task")

    # slot the memory OUT → it drops from live context
    c.patch(f"/api/agent-sessions/{b['id']}/slots", json={"kind": "memory", "ref": "m-0001", "enabled": False})
    ctx2 = "\n".join(m["content"] for m in c.get(f"/api/agent-sessions/{b['id']}/context").json()["messages"])
    assert "abliteration removes refusal" not in ctx2 and "sub findings" in ctx2

    # detach the context slot; the (slotted-out) memory slot stays until detached too
    c.request("DELETE", f"/api/agent-sessions/{b['id']}/slots", params={"kind": "context", "ref": sub["id"]})
    slots = c.get(f"/api/agent-sessions/{b['id']}").json()["slots"]
    assert [s["ref"] for s in slots] == ["m-0001"]
    assert c.delete(f"/api/agent-sessions/{a['id']}").json() == {"removed": a["id"]}
    remaining = {s["id"] for s in c.get("/api/agent-sessions").json()["sessions"]}
    assert a["id"] not in remaining and sub["id"] not in remaining   # subagent cascaded


def test_missing_session_404(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch)
    assert c.get("/api/agent-sessions/ghost").status_code == 404
    assert c.delete("/api/agent-sessions/ghost").status_code == 404
