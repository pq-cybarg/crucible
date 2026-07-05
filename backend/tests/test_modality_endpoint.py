"""/api/abliteration/modality-direction: computes a real direction from supplied embeddings, or
says honestly that it needs them — it never fabricates a direction."""
import numpy as np
from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry


def mkapp(tmp_path):
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"), agent_root=tmp_path))


def _emb(n, dim, shift_dim, shift, seed):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, dim))
    s = np.zeros(dim); s[shift_dim] = shift
    return (base + s).tolist(), rng.standard_normal((n, dim)).tolist()


def test_modality_direction_from_embeddings(tmp_path):
    c = mkapp(tmp_path)
    harmful, benign = _emb(20, 16, 3, 5.0, 0)
    r = c.post("/api/abliteration/modality-direction",
               json={"modality": "image", "harmful_embeddings": harmful, "benign_embeddings": benign}).json()
    assert r["modality"] == "image" and r["dim"] == 16
    assert r["reliable"] is True and r["linearly_encoded"] is True
    assert "held-out" in r["separability_kind"]
    assert len(r["direction"]) == 16
    # plain-language card attached, no jargon, mentions held-out honesty
    assert r["plain"]["technique"] == "modality-direction"
    assert "held-out" in r["plain"]["caveat"].lower()


def test_modality_without_embeddings_is_honest_503(tmp_path):
    c = mkapp(tmp_path)
    r = c.post("/api/abliteration/modality-direction", json={"modality": "audio"})
    assert r.status_code == 503
    assert "no audio embeddings" in r.json()["detail"]
    assert "whisper" in r.json()["detail"]           # tells you how to get them


def test_modality_bad_modality_422(tmp_path):
    c = mkapp(tmp_path)
    r = c.post("/api/abliteration/modality-direction", json={"modality": "smell"})
    assert r.status_code == 422


def test_modality_dim_mismatch_422(tmp_path):
    c = mkapp(tmp_path)
    r = c.post("/api/abliteration/modality-direction",
               json={"modality": "image", "harmful_embeddings": [[1, 2, 3]], "benign_embeddings": [[1, 2]]})
    assert r.status_code == 422
