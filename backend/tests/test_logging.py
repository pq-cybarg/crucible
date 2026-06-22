from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


def test_access_log_does_not_break_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CRUCIBLE_LOG", "1")
    c = TestClient(create_app(registry=Registry(tmp_path / "r.json")))
    assert c.get("/api/health").json() == {"ok": True}
    assert c.get("/api/models").status_code == 200
