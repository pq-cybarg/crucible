from __future__ import annotations
# Import Ollama's downloaded models as first-class Crucible models. Ollama stores each model
# as a content-addressed GGUF blob under ~/.ollama/models/blobs/, described by a manifest.
# Rather than only proxying chat through Ollama, we read the manifest, find the GGUF blob, and
# register it as a local GGUF model — so it becomes fully ours: uncensorable (GGUF abliteration),
# editable, quantizable, retrainable, and servable via llama.cpp. Pure filesystem parsing.
import json
import os
import re
from pathlib import Path

MODEL_MEDIATYPE = "application/vnd.ollama.image.model"


def ollama_root(root: str | None = None) -> Path:
    if root:
        return Path(root)
    return Path(os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama" / "models")))


def _blob_path(root: Path, digest: str) -> Path:
    return root / "blobs" / digest.replace(":", "-")


def list_ollama_models(root: str | None = None) -> list[dict]:
    """Every locally-pulled Ollama model with a GGUF blob: name, blob path, existence, size."""
    base = ollama_root(root)
    mdir = base / "manifests"
    out: list[dict] = []
    if not mdir.exists():
        return out
    for manifest in mdir.rglob("*"):
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        layer = next((ly for ly in (data.get("layers") or [])
                      if isinstance(ly, dict) and ly.get("mediaType") == MODEL_MEDIATYPE), None)
        if not layer:
            continue
        blob = _blob_path(base, layer["digest"])
        parts = manifest.relative_to(mdir).parts       # registry/namespace/model/tag
        name = ("/".join(parts[1:-1]) + ":" + parts[-1]) if len(parts) >= 3 else "/".join(parts)
        out.append({"name": name, "gguf_path": str(blob), "exists": blob.is_file(),
                    "size": int(layer.get("size", 0))})
    return out


def find_ollama_model(name: str, root: str | None = None) -> dict:
    """Locate one Ollama model by name (raises if missing or its blob is absent)."""
    m = next((x for x in list_ollama_models(root) if x["name"] == name), None)
    if m is None:
        raise KeyError(name)
    if not m["exists"]:
        raise FileNotFoundError(m["gguf_path"])
    return m


def registry_id_for(name: str) -> str:
    """A safe registry id from an Ollama model name (llama3:latest -> ollama-llama3-latest)."""
    return "ollama-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
