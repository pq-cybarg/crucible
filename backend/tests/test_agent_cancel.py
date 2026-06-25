from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


class Adapter:
    num_layers = 2
    def generate_chat(self, messages, n=256):
        return "ok"


def mkapp(tmp_path):
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"),
                                 agent_root=tmp_path, abliteration_adapter=Adapter()))


def test_cancel_endpoint_records_run(tmp_path):
    r = mkapp(tmp_path).post("/api/agent/cancel", json={"run_id": "abc"})
    assert r.status_code == 200 and r.json()["cancelled"] == "abc"


def test_run_with_run_id_completes(tmp_path):
    c = mkapp(tmp_path)
    body = {"messages": [{"role": "user", "content": "hi"}],
            "permissions": {"default": "allow", "modes": {}}, "run_id": "run-1"}
    resp = c.post("/api/agent/run", json=body)
    assert resp.status_code == 200 and "data:" in resp.text


def test_precancel_run_emits_cancelled(tmp_path):
    c = mkapp(tmp_path)
    # mark the run cancelled BEFORE starting -> the stream should emit a cancellation
    c.post("/api/agent/cancel", json={"run_id": "run-x"})
    body = {"messages": [{"role": "user", "content": "hi"}],
            "permissions": {"default": "allow", "modes": {}}, "run_id": "run-x"}
    resp = c.post("/api/agent/run", json=body)
    assert "cancelled by operator" in resp.text
