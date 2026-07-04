"""End-to-end: the spawn_agent tool is actually registered and REACHABLE through the real agent
loop. A stub adapter makes the parent emit a ReAct spawn_agent action; the sub-agent (same loop,
fresh context) answers; the parent merges it. Proves the recursive wiring, not just the unit math."""
from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry


class SpawnyAdapter:
    """ReAct feeds a tool result back as a user 'Observation:' message. So: parent's first turn
    (plain task) -> emit a spawn_agent action; a turn whose latest user msg is an Observation ->
    finalize by merging it; a sub-agent (its first task starts 'CHILD') -> just answer."""
    num_layers = 2

    def generate_chat(self, messages, n=256):
        users = [m.get("content", "") for m in messages if m.get("role") == "user"]
        last_user = users[-1] if users else ""
        first_user = users[0] if users else ""
        if last_user.startswith("Observation:"):
            return "Final Answer: parent merged <" + last_user + ">"
        if first_user.startswith("CHILD"):
            return "Final Answer: child handled " + first_user
        return 'Action: spawn_agent\nAction Input: {"task": "CHILD subtask"}'


def mkapp(tmp_path, adapter):
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"),
                                 agent_root=tmp_path, abliteration_adapter=adapter))


def test_swarm_agent_can_actually_spawn_a_subagent(tmp_path):
    c = mkapp(tmp_path, SpawnyAdapter())
    r = c.post("/api/agent/swarm",
               json={"tasks": ["delegate this"], "spawn_depth": 1, "spawn_total": 4}).json()
    assert r["n"] == 1 and r["succeeded"] == 1
    # the parent's final answer embeds the CHILD sub-agent's result -> the tool really ran
    assert "child handled" in r["combined"]
    assert "CHILD subtask" in r["combined"]


def test_swarm_spawn_can_be_disabled(tmp_path):
    # with spawn_depth=0 the tool isn't registered; the parent's spawn action finds no such tool
    c = mkapp(tmp_path, SpawnyAdapter())
    r = c.post("/api/agent/swarm",
               json={"tasks": ["delegate this"], "spawn_depth": 0}).json()
    assert r["n"] == 1
    assert "child handled" not in r["combined"]
