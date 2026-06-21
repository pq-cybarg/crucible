from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


def test_flow_requires_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    c = TestClient(create_app(registry=Registry(tmp_path / "r.json")))
    assert c.post("/api/abliteration/flow", json={"base_id": "x"}).status_code == 503
