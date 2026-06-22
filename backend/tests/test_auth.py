from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


def app(tmp_path, monkeypatch, token=None):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    if token is not None:
        monkeypatch.setenv("CRUCIBLE_API_TOKEN", token)
    else:
        monkeypatch.delenv("CRUCIBLE_API_TOKEN", raising=False)
    return TestClient(create_app(registry=Registry(tmp_path / "r.json")))


def test_open_when_no_token(tmp_path, monkeypatch):
    c = app(tmp_path, monkeypatch)
    assert c.get("/api/models").status_code == 200


def test_blocks_without_token_header(tmp_path, monkeypatch):
    c = app(tmp_path, monkeypatch, token="s3cret")
    assert c.get("/api/models").status_code == 401
    assert c.get("/api/guardrails/presets").status_code == 401


def test_allows_with_bearer(tmp_path, monkeypatch):
    c = app(tmp_path, monkeypatch, token="s3cret")
    assert c.get("/api/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/api/models", headers={"X-Crucible-Token": "s3cret"}).status_code == 200


def test_health_always_open(tmp_path, monkeypatch):
    c = app(tmp_path, monkeypatch, token="s3cret")
    assert c.get("/api/health").status_code == 200


def test_wrong_token_blocked(tmp_path, monkeypatch):
    c = app(tmp_path, monkeypatch, token="s3cret")
    assert c.get("/api/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
