from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


def client(tmp_path):
    return TestClient(create_app(Registry(tmp_path / "registry.json")))


def test_health(tmp_path):
    assert client(tmp_path).get("/api/health").json() == {"ok": True}


def test_create_list_model(tmp_path):
    c = client(tmp_path)
    body = {"id": "glm32b", "name": "GLM-4-32B", "base_id": None,
            "path": "/m/glm32b.gguf", "quant": "Q4_K_M", "kind": "base",
            "endpoint": None, "created": "2026-06-19", "notes": ""}
    r = c.post("/api/models", json=body)
    assert r.status_code == 201
    assert c.get("/api/models").json()[0]["id"] == "glm32b"


def test_duplicate_returns_409(tmp_path):
    c = client(tmp_path)
    body = {"id": "a", "name": "a", "base_id": None, "path": "/m/a.gguf",
            "quant": "Q4_K_M", "kind": "base", "endpoint": None,
            "created": "2026-06-19", "notes": ""}
    c.post("/api/models", json=body)
    assert c.post("/api/models", json=body).status_code == 409


def test_lineage_404(tmp_path):
    assert client(tmp_path).get("/api/models/nope/lineage").status_code == 404
