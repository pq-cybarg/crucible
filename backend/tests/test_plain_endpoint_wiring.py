"""End-to-end: an interpretability endpoint actually ATTACHES the plain-language card (not just
the unit-level explainers). The composition route needs only a GGUF file — no torch — so it's the
one we can exercise fully in CI to prove the wiring, and it covers the with_plain() integration
that every other analysis route shares."""
import struct

import numpy as np
from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


def _write_min_gguf(path, name, arr):
    out, inn = arr.shape

    def gstr(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    buf = bytearray()
    buf += struct.pack("<I", 0x46554747) + struct.pack("<I", 3)
    buf += struct.pack("<Q", 1) + struct.pack("<Q", 1)
    buf += gstr("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", 32)
    buf += gstr(name) + struct.pack("<I", 2) + struct.pack("<Q", inn) + struct.pack("<Q", out)
    buf += struct.pack("<I", 0) + struct.pack("<Q", 0)
    buf += b"\x00" * ((-len(buf)) % 32)
    buf += arr.astype("<f4").tobytes()
    path.write_bytes(buf)


def test_composition_endpoint_attaches_plain_card(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    p = tmp_path / "m.gguf"
    W = np.zeros((8, 4), dtype=np.float32)
    _write_min_gguf(p, "blk.0.attn_output.o_proj.weight", W)   # 'blk.' -> language_model part

    c = TestClient(create_app(registry=Registry(tmp_path / "r.json")))
    r = c.post("/api/abliteration/composition", json={"gguf_path": str(p)})
    assert r.status_code == 200, r.text
    body = r.json()

    # the technical fields are still there ...
    assert "parts" in body and body["n_tensors"] == 1
    # ... AND the plain-language card is attached, fully populated, tagged with the technique.
    plain = body["plain"]
    assert plain["technique"] == "composition"
    for field in ("headline", "what_it_is", "what_we_found", "what_it_means", "caveat"):
        assert isinstance(plain[field], str) and plain[field].strip()
