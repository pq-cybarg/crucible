from __future__ import annotations
# Quantization + quantization analysis. Part of the full pipeline: after editing weights you
# ship a quantized model, and you want to SEE what the quantization costs. This measures the
# per-tensor fidelity of a target quant type (dequantize(quantize(W)) vs W) and can
# requantize between the directly-supported types. Full K-quant requant is out of scope
# (complex super-blocks) — for those, quantize from HF with llama.cpp's converter.
import numpy as np
from numpy.typing import ArrayLike

from crucible.weights.gguf_edit import DIRECT, dequantize, quantize


def quant_error(W: ArrayLike, dtype: str) -> float:
    """Relative error introduced by quantizing W to `dtype` and back (0 = lossless)."""
    Wf = np.asarray(W, dtype=np.float64).ravel()
    n = Wf.size
    if dtype == "Q8_0":
        n -= n % 32                      # Q8_0 works in blocks of 32
        if n == 0:
            return 0.0
        Wf = Wf[:n]
    back = dequantize(quantize(Wf, dtype), dtype, n)
    denom = float(np.linalg.norm(Wf)) or 1.0
    return float(np.linalg.norm(back - Wf) / denom)


def quantize_matrix(W: ArrayLike, dtype: str) -> dict:
    """Quantize a matrix to `dtype`; report byte size, compression ratio, and fidelity."""
    Wf = np.asarray(W, dtype=np.float32)
    raw = quantize(Wf.ravel(), dtype)
    orig_bytes = Wf.size * 4              # vs F32
    return {"dtype": dtype, "bytes": len(raw), "orig_f32_bytes": orig_bytes,
            "compression": round(orig_bytes / max(1, len(raw)), 3),
            "error": round(quant_error(Wf, dtype), 6),
            "fidelity": round(1.0 - quant_error(Wf, dtype), 6)}


def requantize(data: bytes, from_dtype: str, to_dtype: str, n: int) -> bytes:
    """Convert a tensor's bytes from one directly-supported type to another (via F32)."""
    if from_dtype not in DIRECT or to_dtype not in DIRECT:
        raise ValueError(f"requantize supports {sorted(DIRECT)} only")
    return quantize(dequantize(data, from_dtype, n), to_dtype)


def quantization_report(named_matrices: dict, dtype: str) -> dict:
    """Fidelity report for quantizing a set of {name: matrix} to `dtype`."""
    if dtype not in DIRECT:
        return {"dtype": dtype, "supported": False,
                "note": f"{dtype} is a K-quant / not directly supported — quantize from HF with llama.cpp",
                "matrices": []}
    rows = [{"name": name, **quantize_matrix(W, dtype)} for name, W in named_matrices.items()]
    mean_fid = sum(r["fidelity"] for r in rows) / len(rows) if rows else 0.0
    return {"dtype": dtype, "supported": True, "n_matrices": len(rows),
            "mean_fidelity": round(mean_fid, 6), "matrices": rows}
