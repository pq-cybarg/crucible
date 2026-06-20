import re
from collections import Counter

_BLK = re.compile(r"blk\.(\d+)\.")


def layer_of(name: str) -> int:
    m = _BLK.search(name)
    return int(m.group(1)) if m else -1


def group_by_layer(tensors: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for t in tensors:
        groups.setdefault(layer_of(t["name"]), []).append(t)
    return groups


def summarize(parsed: dict) -> dict:
    tensors = parsed["tensors"]
    dtypes = Counter(t["dtype"] for t in tensors)
    groups = group_by_layer(tensors)
    n_layers = len([k for k in groups if k >= 0])
    return {
        "n_tensors": len(tensors),
        "total_params": sum(t["n_params"] for t in tensors),
        "n_layers": n_layers,
        "dtypes": dict(dtypes),
        "architecture": parsed["metadata"].get("general.architecture"),
    }
