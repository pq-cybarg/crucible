from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


class SolverAdapter:
    num_layers = 2
    def generate_chat(self, messages, n=128):
        return "Sure — here's a complete answer."   # never refuses


def mkapp(tmp_path, adapter=None):
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"),
                                 agent_root=tmp_path, abliteration_adapter=adapter))


def test_list_safety_suites(tmp_path):
    s = mkapp(tmp_path).get("/api/evals/safety-suites").json()
    assert "xstest_overrefusal" in s and s["harmbench"]["bundled"] is False


def test_safety_suite_needs_a_model(tmp_path):
    r = mkapp(tmp_path).post("/api/evals/safety-suite", json={"suite": "capability_control"})
    assert r.status_code == 503


def test_safety_suite_overrefusal_passes_when_model_answers(tmp_path):
    c = mkapp(tmp_path, SolverAdapter())
    r = c.post("/api/evals/safety-suite", json={"suite": "xstest_overrefusal"})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "over_refusal"
    assert body["over_refusal_rate"] == 0.0     # an answering model doesn't over-refuse
    assert body["pass_rate"] == 1.0


def test_safety_suite_unknown_404_and_missing_path_409(tmp_path):
    c = mkapp(tmp_path, SolverAdapter())
    assert c.post("/api/evals/safety-suite", json={"suite": "ghost"}).status_code == 404
    assert c.post("/api/evals/safety-suite", json={"suite": "harmbench"}).status_code == 409


def test_contamination_endpoint(tmp_path):
    c = mkapp(tmp_path)
    ref = "the quick brown fox jumps over the lazy dog at dawn every morning today now"
    r = c.post("/api/evals/contamination", json={"candidate": ref, "reference": ref, "n": 5}).json()
    assert r["contaminated"] is True and r["score"] == 1.0


def test_passk_endpoint(tmp_path):
    c = mkapp(tmp_path)
    r = c.post("/api/evals/passk", json={"per_task": [[5, 5], [5, 0]], "k": 1}).json()
    assert r["pass_at_k"] == 0.5
    assert r["per_task"] == [1.0, 0.0]
