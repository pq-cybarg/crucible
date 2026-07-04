from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry


class Adapter:
    num_layers = 2
    def generate_chat(self, messages, n=256):
        task = messages[-1]["content"] if messages else ""
        return f"done: {task}"


def mkapp(tmp_path, adapter=None):
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"),
                                 agent_root=tmp_path, abliteration_adapter=adapter))


def test_swarm_runs_subagent_per_task(tmp_path):
    c = mkapp(tmp_path, Adapter())
    r = c.post("/api/agent/swarm", json={"tasks": ["research A", "research B"]}).json()
    assert r["n"] == 2 and r["succeeded"] == 2
    assert "research A" in r["combined"] and "research B" in r["combined"]


def test_swarm_needs_a_model(tmp_path):
    r = mkapp(tmp_path).post("/api/agent/swarm", json={"tasks": ["x"]})
    assert r.status_code == 503
