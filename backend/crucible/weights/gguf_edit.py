from __future__ import annotations
# Direct GGUF abliteration. Instead of round-tripping through HF safetensors, edit the
# quantized GGUF the operator actually runs — in place, tensor by tensor. We dequantize a
# writing matrix, subtract the rank-1 refusal projection (the same surgical cut as HF
# abliteration), requantize to the SAME type (so byte length is unchanged), and patch the
# bytes back at their original offset — the rest of the file is untouched.
#
# Supported directly: F32, F16, BF16, Q8_0 (clean, well-defined layouts). K-quants (Q4_K,
# Q6_K, …) use complex super-block layouts; editing those in place is error-prone, so we
# report them as unsupported and point at the HF-abliterate → re-quantize path.
import struct

import numpy as np

QK8_0 = 32                      # Q8_0 block size
Q8_0_BYTES = 2 + QK8_0          # f16 scale + 32 int8
DIRECT = {"F32", "F16", "BF16", "Q8_0"}


def _f16_to_f32(u16: np.ndarray) -> np.ndarray:
    return u16.view(np.float16).astype(np.float32)


def dequantize(data: bytes, dtype: str, n: int) -> np.ndarray:
    """Bytes -> float32 array of length n, for the directly-supported types."""
    if dtype == "F32":
        return np.frombuffer(data, dtype="<f4", count=n).astype(np.float32)
    if dtype == "F16":
        return np.frombuffer(data, dtype="<f2", count=n).astype(np.float32)
    if dtype == "BF16":
        u = np.frombuffer(data, dtype="<u2", count=n).astype(np.uint32)
        return (u << 16).view(np.float32)
    if dtype == "Q8_0":
        nb = n // QK8_0
        raw = np.frombuffer(data, dtype=np.uint8, count=nb * Q8_0_BYTES).reshape(nb, Q8_0_BYTES)
        scales = _f16_to_f32(raw[:, :2].copy().view(np.uint16).reshape(nb))
        q = raw[:, 2:].view(np.int8).astype(np.float32)      # (nb, 32)
        return (q * scales[:, None]).reshape(n)
    raise ValueError(f"dequantize: unsupported type {dtype}")


def quantize(arr: np.ndarray, dtype: str) -> bytes:
    """Float32 array -> bytes of the given type (same layout dequantize expects)."""
    a = np.asarray(arr, dtype=np.float32).ravel()
    if dtype == "F32":
        return a.astype("<f4").tobytes()
    if dtype == "F16":
        return a.astype("<f2").tobytes()
    if dtype == "BF16":
        u = a.view(np.uint32)
        return ((u >> 16).astype(np.uint16)).astype("<u2").tobytes()
    if dtype == "Q8_0":
        n = a.size
        nb = n // QK8_0
        blocks = a[: nb * QK8_0].reshape(nb, QK8_0)
        amax = np.abs(blocks).max(axis=1)
        d = amax / 127.0
        d_safe = np.where(d == 0, 1.0, d)
        q = np.round(blocks / d_safe[:, None]).clip(-127, 127).astype(np.int8)
        out = bytearray()
        d16 = d.astype(np.float16).view(np.uint16)
        for i in range(nb):
            out += struct.pack("<H", int(d16[i]))
            out += q[i].tobytes()
        return bytes(out)
    raise ValueError(f"quantize: unsupported type {dtype}")


def edit_matrix(W: np.ndarray, r: np.ndarray, mode: str = "unalign", coef: float = 1.0) -> np.ndarray:
    """Edit the refusal component of a writing matrix. mode='unalign' removes it
    (W - coef*r(rT W)), 'realign' restores/strengthens it (W + coef*r(rT W))."""
    r = np.asarray(r, dtype=np.float64)
    r = r / (float(np.linalg.norm(r)) or 1.0)
    Wf = np.asarray(W, dtype=np.float64)
    sign = 1.0 if mode == "realign" else -1.0
    return (Wf + sign * coef * np.outer(r, r @ Wf)).astype(np.float32)


def orthogonalize_matrix(W: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Surgical cut: W' = W - r (rᵀ W), removing only the rank-1 component of W along the
    (unit) refusal direction r. r's length must equal W's output dimension (rows)."""
    r = np.asarray(r, dtype=np.float64)
    r = r / (float(np.linalg.norm(r)) or 1.0)
    Wf = np.asarray(W, dtype=np.float64)
    return (Wf - np.outer(r, r @ Wf)).astype(np.float32)


def tensor_matrix_shape(dims: list[int]) -> tuple[int, int]:
    """GGUF stores dims fastest-first (ne0 = in, ne1 = out); the row-major data is (out, in)."""
    if len(dims) != 2:
        raise ValueError("expected a 2-D tensor")
    ne0, ne1 = int(dims[0]), int(dims[1])
    return ne1, ne0            # (out, in)


def abliterate_gguf(path: str, direction, name_filter=("o_proj", "down_proj"),
                    dry_run: bool = False, mode: str = "unalign", coef: float = 1.0) -> dict:
    """Abliterate a GGUF in place (edits the file unless dry_run). Patches 2-D writing
    matrices whose name contains one of name_filter, when their type is directly editable."""
    from crucible.weights.gguf_reader import parse_gguf
    parsed = parse_gguf(path)
    r = np.asarray(direction, dtype=np.float64)
    edited, skipped = [], []
    patches: list[tuple[int, bytes]] = []
    for t in parsed["tensors"]:
        if len(t["shape"]) != 2 or not any(k in t["name"] for k in name_filter):
            continue
        out_dim, in_dim = tensor_matrix_shape(t["shape"])
        if t["dtype"] not in DIRECT:
            skipped.append({"name": t["name"], "dtype": t["dtype"], "reason": "quant not directly editable"})
            continue
        if out_dim != r.shape[0]:
            skipped.append({"name": t["name"], "dtype": t["dtype"], "reason": f"dim {out_dim} != direction {r.shape[0]}"})
            continue
        with open(path, "rb") as f:
            f.seek(t["abs_offset"])
            nbytes = _tensor_nbytes(t["dtype"], t["n_params"])
            raw = f.read(nbytes)
        W = dequantize(raw, t["dtype"], t["n_params"]).reshape(out_dim, in_dim)
        W2 = edit_matrix(W, r, mode, coef)
        new_bytes = quantize(W2, t["dtype"])
        if len(new_bytes) != nbytes:
            skipped.append({"name": t["name"], "dtype": t["dtype"], "reason": "requant size mismatch"})
            continue
        patches.append((t["abs_offset"], new_bytes))
        edited.append({"name": t["name"], "dtype": t["dtype"], "shape": [out_dim, in_dim]})
    if not dry_run and patches:
        with open(path, "r+b") as f:
            for off, b in patches:
                f.seek(off)
                f.write(b)
    return {"edited": edited, "skipped": skipped, "n_edited": len(edited),
            "n_skipped": len(skipped), "dry_run": dry_run}


def _tensor_nbytes(dtype: str, n: int) -> int:
    if dtype == "F32":
        return n * 4
    if dtype in ("F16", "BF16"):
        return n * 2
    if dtype == "Q8_0":
        return (n // QK8_0) * Q8_0_BYTES
    raise ValueError(f"nbytes: unsupported type {dtype}")
