import struct

from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Model, Registry
from crucible.weights.gguf_reader import GGUF_MAGIC


def _gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def make_gguf(path):
    blob = struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 1)
    blob += _gstr("general.architecture") + struct.pack("<I", 8) + _gstr("glm")
    blob += _gstr("blk.0.attn_q.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 64, 64) + struct.pack("<I", 1) + struct.pack("<Q", 0)
    path.write_bytes(blob)


def test_weights_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    gguf = tmp_path / "m.gguf"
    make_gguf(gguf)
    reg = Registry(tmp_path / "r.json")
    reg.register(Model(id="m", name="m", base_id=None, path=str(gguf), quant="F16",
                       kind="base", endpoint=None, created="2026-06-20", notes=""))
    c = TestClient(create_app(registry=reg))
    body = c.get("/api/weights/m").json()
    assert body["summary"]["architecture"] == "glm"
    assert body["summary"]["n_tensors"] == 1
    assert body["tensors"][0]["name"] == "blk.0.attn_q.weight"


def test_weights_missing_file_404(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    reg = Registry(tmp_path / "r.json")
    reg.register(Model(id="m", name="m", base_id=None, path="/nope/x.gguf", quant="F16",
                       kind="base", endpoint=None, created="2026-06-20", notes=""))
    c = TestClient(create_app(registry=reg))
    assert c.get("/api/weights/m").status_code == 404
