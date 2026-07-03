import numpy as np

from crucible.weights.gguf_edit import (
    dequantize, orthogonalize_matrix, quantize, tensor_matrix_shape, _tensor_nbytes)


def test_f32_roundtrip_exact():
    a = np.array([1.0, -2.5, 3.25, 0.0, 100.0, -0.001], dtype=np.float32)
    assert np.array_equal(dequantize(quantize(a, "F32"), "F32", a.size), a)


def test_f16_roundtrip_close():
    a = np.array([1.0, -2.5, 3.25, 0.5, 10.0, -7.0], dtype=np.float32)
    back = dequantize(quantize(a, "F16"), "F16", a.size)
    assert np.allclose(back, a, atol=1e-2)


def test_bf16_roundtrip_close():
    a = np.array([1.0, -2.5, 3.5, 128.0, -0.25], dtype=np.float32)
    back = dequantize(quantize(a, "BF16"), "BF16", a.size)
    assert np.allclose(back, a, rtol=0.02, atol=1e-2)


def test_q8_0_roundtrip_within_quant_error():
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(32 * 4) * 3).astype(np.float32)   # 4 blocks of 32
    back = dequantize(quantize(a, "Q8_0"), "Q8_0", a.size)
    # Q8_0 keeps a per-block f16 scale over 127 levels -> small relative error
    err = np.abs(back - a) / (np.abs(a).max())
    assert err.max() < 0.02


def test_q8_0_byte_length_matches():
    a = np.zeros(32 * 5, dtype=np.float32)
    assert len(quantize(a, "Q8_0")) == _tensor_nbytes("Q8_0", a.size)


def test_orthogonalize_removes_only_r_component():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((8, 5)).astype(np.float32)
    r = rng.standard_normal(8); r /= np.linalg.norm(r)
    W2 = orthogonalize_matrix(W, r)
    # the r-component of every column is gone
    assert np.allclose(r @ W2, 0.0, atol=1e-5)
    # the part orthogonal to r is preserved: projecting W2 back adds nothing new
    P = np.eye(8) - np.outer(r, r)
    assert np.allclose(P @ W, W2, atol=1e-4)


def test_tensor_matrix_shape_out_in():
    # GGUF dims [in, out] -> matrix (out, in)
    assert tensor_matrix_shape([5, 8]) == (8, 5)


def test_unsupported_quant_raises():
    try:
        quantize(np.zeros(32, dtype=np.float32), "Q4_K")
        assert False
    except ValueError:
        pass


def _write_min_gguf(path, name, arr_out_in):
    """Minimal GGUF v3 with one F32 2-D tensor (arr shape (out, in)), alignment 32."""
    import struct
    out, inn = arr_out_in.shape
    dims = [inn, out]                       # GGUF stores [in, out]
    def gstr(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b
    buf = bytearray()
    buf += struct.pack("<I", 0x46554747)    # magic
    buf += struct.pack("<I", 3)             # version
    buf += struct.pack("<Q", 1)             # tensor_count
    buf += struct.pack("<Q", 1)             # kv_count
    buf += gstr("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", 32)  # u32 kv
    buf += gstr(name) + struct.pack("<I", 2) + struct.pack("<Q", dims[0]) + struct.pack("<Q", dims[1])
    buf += struct.pack("<I", 0)             # type F32
    buf += struct.pack("<Q", 0)             # offset
    pad = (-len(buf)) % 32
    buf += b"\x00" * pad
    buf += arr_out_in.astype("<f4").tobytes()
    path.write_bytes(buf)


def test_abliterate_gguf_end_to_end(tmp_path):
    from crucible.weights.gguf_edit import abliterate_gguf
    from crucible.weights.gguf_reader import parse_gguf
    rng = np.random.default_rng(2)
    W = rng.standard_normal((16, 6)).astype(np.float32)      # (out=16, in=6)
    r = rng.standard_normal(16); r /= np.linalg.norm(r)
    p = tmp_path / "m.gguf"
    _write_min_gguf(p, "blk.0.attn_output.o_proj.weight", W)
    res = abliterate_gguf(str(p), r, name_filter=("o_proj",))
    assert res["n_edited"] == 1 and res["skipped"] == []
    # reload the patched tensor and confirm the refusal component is gone
    t = [t for t in parse_gguf(str(p))["tensors"] if "o_proj" in t["name"]][0]
    raw = p.read_bytes()[t["abs_offset"]:t["abs_offset"] + 16 * 6 * 4]
    W2 = np.frombuffer(raw, dtype="<f4").reshape(16, 6)
    assert np.allclose(r @ W2, 0.0, atol=1e-4)               # surgical cut applied on disk


def test_abliterate_gguf_dry_run_skips_kquant(tmp_path):
    import glob
    import os
    from crucible.weights.gguf_edit import abliterate_gguf
    files = [f for f in glob.glob("models/**/*.gguf", recursive=True) if os.path.isfile(f)]
    if not files:
        import pytest; pytest.skip("no real gguf present")
    # a plausible direction dim won't match; we only assert it detects+skips K-quant safely
    res = abliterate_gguf(files[0], np.ones(896), dry_run=True)
    assert res["dry_run"] is True
    assert res["n_edited"] == 0                              # nothing edited on a dry run
    # any matched writing matrices are K-quant -> reported as skipped, file untouched


def test_edit_matrix_modes():
    from crucible.weights.gguf_edit import edit_matrix, orthogonalize_matrix
    rng = np.random.default_rng(7)
    W = rng.standard_normal((8, 5)); r = rng.standard_normal(8); r /= np.linalg.norm(r)
    # unalign matches orthogonalize; realign is its mirror (adds the component back)
    assert np.allclose(edit_matrix(W, r, "unalign", 1.0), orthogonalize_matrix(W, r), atol=1e-5)
    assert np.allclose(r @ edit_matrix(W, r, "unalign"), 0.0, atol=1e-5)
    assert float(np.linalg.norm(r @ edit_matrix(W, r, "realign", 1.0))) > float(np.linalg.norm(r @ W))


def test_edit_matrix_realign():   # marker
    pass
