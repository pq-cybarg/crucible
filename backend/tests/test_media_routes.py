from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry


def mkapp(tmp_path, monkeypatch):
    for k in ("EMBED", "IMAGE", "STT", "TTS"):
        monkeypatch.delenv(f"CRUCIBLE_{k}_ENDPOINT", raising=False)
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"), agent_root=tmp_path))


def test_media_routes_503_without_backend(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch)
    assert c.post("/v1/embeddings", json={"input": "hi"}).status_code == 503
    assert c.post("/v1/images/generations", json={"prompt": "a cat"}).status_code == 503
    assert c.post("/v1/audio/transcriptions", json={"file": "a.wav"}).status_code == 503
    assert c.post("/v1/audio/speech", json={"input": "hello"}).status_code == 503


def test_media_status_route_reports_capabilities(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch)   # all backends unset
    st = c.get("/api/media/status").json()
    assert st["n_configured"] == 0 and st["n_total"] == 4
    assert set(st["backends"]) == {"image", "stt", "tts", "embed"}
    assert all(b["configured"] is False for b in st["backends"].values())
    assert "brokered" in st["note"]
