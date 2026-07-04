import json
from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry
from crucible.ollama_import import MODEL_MEDIATYPE


def _store(tmp_path):
    man = tmp_path / "manifests" / "reg" / "library" / "tiny" / "latest"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps({"layers": [
        {"mediaType": MODEL_MEDIATYPE, "digest": "sha256:deadbeef", "size": 4}]}))
    blob = tmp_path / "blobs" / "sha256-deadbeef"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"GGUF")
    return tmp_path


def mkapp(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODELS", str(_store(tmp_path / "olla")))
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"), agent_root=tmp_path))


def test_list_ollama(tmp_path, monkeypatch):
    rows = mkapp(tmp_path, monkeypatch).get("/api/models/ollama").json()
    assert any(r["name"] == "library/tiny:latest" for r in rows)
    assert rows[0]["suggested_id"].startswith("ollama-")


def test_import_ollama_registers_gguf(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch)
    r = c.post("/api/models/import-ollama", json={"name": "library/tiny:latest"})
    assert r.status_code == 201
    m = r.json()
    assert m["quant"] == "gguf" and m["path"].endswith("sha256-deadbeef")
    # importing again is idempotent
    assert c.post("/api/models/import-ollama", json={"name": "library/tiny:latest"}).status_code in (201,)


def test_import_unknown_404(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch)
    assert c.post("/api/models/import-ollama", json={"name": "ghost:latest"}).status_code == 404
