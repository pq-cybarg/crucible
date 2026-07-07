"""Model tool-capability probe — the forge uses it to auto-enable compatibility mode in plain words."""
from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Model, Registry


def mkapp(tmp_path):
    reg = Registry(tmp_path / "r.json")
    reg.register(Model(id="no-endpoint", name="n", base_id=None, path="remote::x", quant="remote",
                       kind="base", endpoint=None, created="2026-07-07", notes=""))
    return TestClient(create_app(registry=reg, agent_root=tmp_path))


def test_tool_support_unknown_model_404(tmp_path):
    assert mkapp(tmp_path).get("/api/models/ghost/tool-support").status_code == 404


def test_tool_support_unknown_when_offline(tmp_path):
    # no live endpoint -> we don't launch a model just to probe; report unknown honestly
    r = mkapp(tmp_path).get("/api/models/no-endpoint/tool-support").json()
    assert r["supports_tools"] is None and "not online" in r["reason"]


def test_forget_and_repoint_model(tmp_path):
    c = mkapp(tmp_path)
    # re-point a dead entry at a live endpoint -> re-enable path
    m = c.post("/api/models/no-endpoint/endpoint", json={"endpoint": "http://localhost:11434/"}).json()
    assert m["endpoint"] == "http://localhost:11434"        # trailing slash trimmed
    # forget removes the entry
    assert c.delete("/api/models/no-endpoint").json() == {"removed": "no-endpoint"}
    assert c.get("/api/models/no-endpoint/tool-support").status_code == 404
    # forgetting an unknown id -> 404
    assert c.delete("/api/models/ghost").status_code == 404
