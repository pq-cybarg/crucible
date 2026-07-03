import numpy as np

from crucible.weights.quantize import (
    quant_error, quantize_matrix, quantization_report, requantize)


def test_f16_lower_error_than_q8_but_both_small():
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((32, 32)) * 2).astype(np.float32)
    e_f16 = quant_error(W, "F16")
    e_q8 = quant_error(W, "Q8_0")
    assert e_f16 < 0.01
    assert e_q8 < 0.05
    assert e_f16 <= e_q8            # F16 keeps more precision than Q8_0


def test_f32_is_lossless():
    W = np.array([[1.5, -2.25], [3.0, 0.0]], dtype=np.float32)
    assert quant_error(W, "F32") == 0.0


def test_quantize_matrix_compression_and_fidelity():
    W = np.ones((32, 32), dtype=np.float32)
    r = quantize_matrix(W, "Q8_0")
    assert r["compression"] > 3.5              # Q8_0 ~ 4x smaller than F32
    assert r["fidelity"] > 0.99
    assert r["bytes"] < r["orig_f32_bytes"]


def test_requantize_between_supported_types():
    rng = np.random.default_rng(1)
    W = (rng.standard_normal(64)).astype(np.float32)
    from crucible.weights.gguf_edit import quantize, dequantize
    f16_bytes = quantize(W, "F16")
    q8_bytes = requantize(f16_bytes, "F16", "Q8_0", W.size)
    back = dequantize(q8_bytes, "Q8_0", W.size)
    assert np.allclose(back, W, atol=0.1)


def test_requantize_rejects_kquant():
    try:
        requantize(b"", "F16", "Q4_K", 32)
        assert False
    except ValueError:
        pass


def test_report_flags_unsupported_kquant():
    rep = quantization_report({"w": np.ones((4, 4), dtype=np.float32)}, "Q4_K")
    assert rep["supported"] is False and rep["matrices"] == []


def test_report_supported_type():
    rep = quantization_report({"a": np.ones((32, 32), dtype=np.float32),
                               "b": np.zeros((32, 32), dtype=np.float32)}, "F16")
    assert rep["supported"] is True and rep["n_matrices"] == 2
    assert rep["mean_fidelity"] > 0.99
