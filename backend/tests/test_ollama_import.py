import json

from crucible.ollama_import import (
    find_ollama_model, list_ollama_models, registry_id_for, MODEL_MEDIATYPE)


def _fake_ollama(root, name="llama3", tag="latest", with_blob=True):
    """Build a minimal Ollama store: a manifest referencing a GGUF blob."""
    digest = "sha256:abc123"
    man = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps({
        "layers": [
            {"mediaType": "application/vnd.ollama.image.license", "digest": "sha256:lic", "size": 10},
            {"mediaType": MODEL_MEDIATYPE, "digest": digest, "size": 4096},
        ]
    }))
    if with_blob:
        blob = root / "blobs" / "sha256-abc123"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"GGUF fake weights")
    return root


def test_list_finds_model_and_blob(tmp_path):
    _fake_ollama(tmp_path)
    models = list_ollama_models(str(tmp_path))
    assert len(models) == 1
    m = models[0]
    assert m["name"] == "library/llama3:latest"
    assert m["exists"] is True and m["size"] == 4096
    assert m["gguf_path"].endswith("sha256-abc123")


def test_list_empty_when_no_store(tmp_path):
    assert list_ollama_models(str(tmp_path / "nope")) == []


def test_find_missing_raises(tmp_path):
    _fake_ollama(tmp_path)
    try:
        find_ollama_model("ghost", str(tmp_path))
        assert False
    except KeyError:
        pass


def test_find_reports_absent_blob(tmp_path):
    _fake_ollama(tmp_path, with_blob=False)
    try:
        find_ollama_model("library/llama3:latest", str(tmp_path))
        assert False
    except FileNotFoundError:
        pass


def test_registry_id_sanitizes():
    assert registry_id_for("llama3:latest") == "ollama-llama3-latest"
    assert registry_id_for("qwen2.5:0.5b") == "ollama-qwen2-5-0-5b"


def test_ignores_manifests_without_model_layer(tmp_path):
    man = tmp_path / "manifests" / "reg" / "ns" / "novgguf" / "latest"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps({"layers": [{"mediaType": "other", "digest": "sha256:x"}]}))
    assert list_ollama_models(str(tmp_path)) == []
