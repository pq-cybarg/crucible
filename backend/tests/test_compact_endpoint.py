"""The /api/agent/compact endpoint summarizes old turns using a real (stub) model, and the agent
run auto-compacts when over budget. The stub adapter stands in for the summarizing model."""
from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry


class Summarizer:
    num_layers = 2
    def generate_chat(self, messages, n=256):
        return "COMPACT SUMMARY: goal + key facts preserved."


def mkapp(tmp_path, adapter=None):
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"),
                                 agent_root=tmp_path, abliteration_adapter=adapter))


def _long_convo(n=20):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"q{i} " + "x" * 40})
        msgs.append({"role": "assistant", "content": f"a{i} " + "y" * 40})
    return msgs


def test_compact_endpoint_summarizes(tmp_path):
    c = mkapp(tmp_path, Summarizer())
    body = {"messages": _long_convo(20), "keep_recent": 4, "force": True}
    r = c.post("/api/agent/compact", json=body).json()
    assert r["compacted"] is True
    assert "COMPACT SUMMARY" in r["summary"]
    # system prompt + summary + last 4 turns
    assert r["messages"][0]["content"] == "sys"
    assert any("Summary of earlier conversation" in m["content"]
               for m in r["messages"] if m["role"] == "system")
    assert r["stats"]["after_tokens"] < r["stats"]["before_tokens"]
    assert "heuristic" in r["stats"]["token_estimate"]


def test_compact_endpoint_needs_a_model(tmp_path):
    c = mkapp(tmp_path)   # no adapter, no model
    r = c.post("/api/agent/compact", json={"messages": _long_convo(4)})
    assert r.status_code == 503


def test_compact_force_false_passthrough_under_budget(tmp_path):
    c = mkapp(tmp_path, Summarizer())
    r = c.post("/api/agent/compact",
               json={"messages": _long_convo(2), "force": False, "max_tokens": 100000}).json()
    assert r["compacted"] is False
