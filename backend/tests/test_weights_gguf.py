import struct

import pytest

from crucible.weights.gguf_reader import GGUF_MAGIC, parse_gguf


def _gstr(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def build_gguf(tmp_path):
    blob = struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", 3) + struct.pack("<Q", 2) + struct.pack("<Q", 1)
    blob += _gstr("general.architecture") + struct.pack("<I", 8) + _gstr("glm")
    blob += _gstr("blk.0.attn_q.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 4096, 4096) + struct.pack("<I", 1) + struct.pack("<Q", 0)
    blob += _gstr("output.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 4096, 1000) + struct.pack("<I", 14) + struct.pack("<Q", 100)
    p = tmp_path / "m.gguf"
    p.write_bytes(blob)
    return p


def test_parse_gguf(tmp_path):
    parsed = parse_gguf(str(build_gguf(tmp_path)))
    assert parsed["metadata"]["general.architecture"] == "glm"
    names = [t["name"] for t in parsed["tensors"]]
    assert names == ["blk.0.attn_q.weight", "output.weight"]
    t0 = parsed["tensors"][0]
    assert t0["shape"] == [4096, 4096] and t0["dtype"] == "F16" and t0["n_params"] == 4096 * 4096
    assert parsed["tensors"][1]["dtype"] == "Q6_K"


def test_not_gguf_raises(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"NOPE" + b"\x00" * 20)
    with pytest.raises(ValueError):
        parse_gguf(str(p))
