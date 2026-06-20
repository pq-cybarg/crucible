from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


def test_verify_unknown_ids_404(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    c = TestClient(create_app(registry=Registry(tmp_path / "r.json")))
    r = c.post("/api/abliteration/verify", json={"base_id": "nope", "variant_id": "nope2"})
    assert r.status_code == 404
