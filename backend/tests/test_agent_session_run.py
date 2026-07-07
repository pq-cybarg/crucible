"""Running a tab's agent: executes in the tab's cwd with its assembled (slotted) context, streams,
and persists the conversation back to the session."""
from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


class StubModel:
    """A trivial chat model — replies with a fixed line, no tool calls (base case for hybrid_run)."""
    def __call__(self, messages, tools):
        # echo that it saw the loaded context, so we can assert slots reached the model
        saw = "notes-loaded" if any("loaded memory" in m.get("content", "") for m in messages) else "plain"
        return {"role": "assistant", "content": f"done ({saw})"}


def mkapp(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"), agent_root=tmp_path, model=StubModel()))


def test_run_persists_conversation_and_uses_loaded_slots(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch)
    sid = c.post("/api/agent-sessions", json={"title": "work", "cwd": str(tmp_path)}).json()["id"]
    # load a crystallized memory into the tab
    from crucible.memory import MemoryStore
    from crucible.config import get_settings
    MemoryStore(get_settings().data_dir / "memory").crystallize(
        [{"role": "user", "content": "x"}], "abliteration notes", label="notes")
    c.post(f"/api/agent-sessions/{sid}/slots", json={"kind": "memory", "ref": "m-0001"})

    r = c.post(f"/api/agent-sessions/{sid}/run", json={"message": "do the task", "run_id": "t1"})
    assert r.status_code == 200
    body = r.text
    assert "notes-loaded" in body                          # the loaded memory reached the model

    # the tab now holds the real conversation (user + assistant), slots NOT stored in it
    doc = c.get(f"/api/agent-sessions/{sid}").json()
    roles = [m["role"] for m in doc["messages"]]
    assert roles == ["user", "assistant"]
    assert doc["messages"][0]["content"] == "do the task"
    assert doc["messages"][1]["content"].startswith("done")
    assert doc["status"] == "idle"                          # run finished


def test_run_guards(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch)
    sid = c.post("/api/agent-sessions", json={"title": "w", "cwd": "."}).json()["id"]
    assert c.post(f"/api/agent-sessions/{sid}/run", json={"message": "  "}).status_code == 422
    assert c.post("/api/agent-sessions/ghost/run", json={"message": "hi"}).status_code == 404
