"""Hierarchy-profile endpoints + the communicator relay wired through the swarm. A stub adapter
stands in for both the worker and (weaker) communicator model."""
from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry


class Adapter:
    """Worker: on the child's task, emit a spawn action once then finalize with a verbose result.
    Also serves as the 'communicator' summarizer (it just tags the text)."""
    num_layers = 2
    def generate_chat(self, messages, n=256):
        users = [m.get("content", "") for m in messages if m.get("role") == "user"]
        last = users[-1] if users else ""
        if last.startswith("You are a communicator"):
            return "RELAYED: tight summary"
        # any task -> a plain verbose final answer (no tools) so the swarm completes quickly
        return "Final Answer: a long verbose child result with lots of detail " * 3


def mkapp(tmp_path, monkeypatch, adapter=None):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"),
                                 agent_root=tmp_path, abliteration_adapter=adapter))


def test_profiles_crud(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch, Adapter())
    assert any(p["name"] == "flat" for p in c.get("/api/hierarchy/profiles").json()["profiles"])
    saved = c.post("/api/hierarchy/profiles", json={
        "name": "research", "layers": [{"worker": None, "communicator": None}, {"worker": "big", "communicator": "small"}]}).json()
    assert saved["name"] == "research" and saved["layers"][1]["communicator"] == "small"
    assert c.delete("/api/hierarchy/profiles/research").json()["deleted"] is True
    assert c.post("/api/hierarchy/profiles", json={"name": ""}).status_code == 422


class Spawny:
    """Parent spawns a child; child finalizes; when asked as the communicator, relays 'RELAYED-UP'."""
    num_layers = 2
    def generate_chat(self, messages, n=256):
        users = [m.get("content", "") for m in messages if m.get("role") == "user"]
        first = users[0] if users else ""
        last = users[-1] if users else ""
        if last.startswith("You are a communicator"):
            return "RELAYED-UP"
        if first.startswith("CHILD"):
            return "Final Answer: verbose child text " * 5
        if last.startswith("Observation:"):
            return "Final Answer: parent saw <" + last + ">"
        return 'Action: spawn_agent\nAction Input: {"task": "CHILD go"}'


def test_swarm_with_profile_relays_child_result(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    from crucible.registry import Model, Registry
    reg = Registry(tmp_path / "r.json")
    # a registered model with NO endpoint resolves to the loaded adapter -> the communicator uses it
    reg.register(Model(id="comm", name="comm", base_id=None, path="x", quant="Q4",
                       kind="base", endpoint=None, created="2026"))
    from fastapi.testclient import TestClient
    from crucible.app import create_app
    c = TestClient(create_app(registry=reg, agent_root=tmp_path, abliteration_adapter=Spawny()))
    c.post("/api/hierarchy/profiles", json={"name": "p", "layers": [
        {"worker": None, "communicator": None}, {"worker": None, "communicator": "comm"}]})
    r = c.post("/api/agent/swarm", json={"tasks": ["do it"], "spawn_depth": 2, "profile": "p"}).json()
    assert r["succeeded"] == 1
    assert "RELAYED-UP" in r["combined"]     # the child's result was compressed by the communicator
