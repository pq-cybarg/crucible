import struct

GGUF_MAGIC = 0x46554747

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
}

_SCALAR_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
               10: "Q", 11: "q", 12: "d"}


class _Reader:
    def __init__(self, f):
        self.f = f

    def _unpack(self, fmt: str):
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.f.read(size))[0]

    def u32(self) -> int:
        return self._unpack("I")

    def u64(self) -> int:
        return self._unpack("Q")

    def gstring(self) -> str:
        length = self.u64()
        return self.f.read(length).decode("utf-8", "replace")

    def value(self, vtype: int):
        if vtype == 8:  # STRING
            return self.gstring()
        if vtype == 9:  # ARRAY
            elem_type = self.u32()
            count = self.u64()
            return [self.value(elem_type) for _ in range(count)]
        return self._unpack(_SCALAR_FMT[vtype])


def parse_gguf(path: str) -> dict:
    """Parse a GGUF header into metadata + tensor descriptors (no weight data read)."""
    with open(path, "rb") as f:
        r = _Reader(f)
        if r.u32() != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        version = r.u32()
        tensor_count = r.u64()
        kv_count = r.u64()
        metadata: dict = {}
        for _ in range(kv_count):
            key = r.gstring()
            vtype = r.u32()
            metadata[key] = r.value(vtype)
        tensors: list[dict] = []
        for _ in range(tensor_count):
            name = r.gstring()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ttype = r.u32()
            offset = r.u64()
            n_params = 1
            for d in dims:
                n_params *= d
            tensors.append({
                "name": name,
                "shape": dims,
                "dtype": GGML_TYPES.get(ttype, f"type{ttype}"),
                "n_params": n_params,
                "offset": offset,
            })
        return {"version": version, "tensor_count": tensor_count,
                "metadata": metadata, "tensors": tensors}
