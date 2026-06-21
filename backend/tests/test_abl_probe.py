from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


def test_probe_requires_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    c = TestClient(create_app(registry=Registry(tmp_path / "r.json")))
    r = c.post("/api/abliteration/probe", json={"base_id": "x", "layers": [1]})
    assert r.status_code == 503
