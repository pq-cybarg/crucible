"""/api/graph/run end-to-end: model / transform / vote / cascade stages compose into a DAG.
A stub adapter stands in for every model call (model_id=null resolves to it)."""
from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry


class EchoAdapter:
    """Echoes the incoming prompt so we can trace how inputs flow through the graph."""
    num_layers = 2
    def generate_chat(self, messages, n=256):
        return "ECHO:" + (messages[-1]["content"] if messages else "")


def mkapp(tmp_path, adapter=None):
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"),
                                 agent_root=tmp_path, abliteration_adapter=adapter))


def test_graph_linear_model_pipeline(tmp_path):
    c = mkapp(tmp_path, EchoAdapter())
    stages = [
        {"id": "a", "inputs": [], "kind": "model", "config": {"prompt": "step1 {input}"}},
        {"id": "b", "inputs": ["a"], "kind": "model", "config": {"prompt": "step2 {input}"}},
    ]
    r = c.post("/api/graph/run", json={"stages": stages, "initial": "GO"}).json()
    assert r["order"] == ["a", "b"]
    assert "GO" in r["outputs"]["a"]
    assert "step2" in r["outputs"]["b"] and "step1" in r["outputs"]["b"]   # a's output fed into b
    assert list(r["result"]) == ["b"]                                     # terminal stage


def test_graph_vote_stage_merges_fan_out(tmp_path):
    c = mkapp(tmp_path, EchoAdapter())
    # three verifiers all echo the same thing -> majority agreement 1.0
    stages = [
        {"id": "v1", "inputs": [], "kind": "model", "config": {"prompt": "same"}},
        {"id": "v2", "inputs": [], "kind": "model", "config": {"prompt": "same"}},
        {"id": "v3", "inputs": [], "kind": "model", "config": {"prompt": "same"}},
        {"id": "vote", "inputs": ["v1", "v2", "v3"], "kind": "vote", "config": {"strategy": "majority"}},
    ]
    r = c.post("/api/graph/run", json={"stages": stages}).json()
    v = r["outputs"]["vote"]
    assert v["strategy"] == "majority" and v["n"] == 3 and v["agreement"] == 1.0
    assert v["result"] == "ECHO:same"


def test_graph_cascade_escalates(tmp_path):
    c = mkapp(tmp_path, EchoAdapter())
    # acceptor demands a token the echo never contains -> escalate through all models, accepted=False
    stages = [
        {"id": "casc", "inputs": [], "kind": "cascade",
         "config": {"models": [None, None], "prompt": "answer {input}",
                    "accept": {"must_include": ["NOBEL_PRIZE"]}}},
    ]
    r = c.post("/api/graph/run", json={"stages": stages, "initial": "q"}).json()
    casc = r["outputs"]["casc"]
    assert casc["tried"] == ["None", "None"] and casc["escalated"] is True
    assert casc["accepted"] is False


def test_graph_cascade_needs_models(tmp_path):
    c = mkapp(tmp_path, EchoAdapter())
    stages = [{"id": "casc", "inputs": [], "kind": "cascade", "config": {"models": []}}]
    r = c.post("/api/graph/run", json={"stages": stages})
    assert r.status_code == 422


def test_graph_rejects_cycle(tmp_path):
    c = mkapp(tmp_path, EchoAdapter())
    stages = [{"id": "a", "inputs": ["b"], "kind": "transform"},
              {"id": "b", "inputs": ["a"], "kind": "transform"}]
    assert c.post("/api/graph/run", json={"stages": stages}).status_code == 422
