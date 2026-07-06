from __future__ import annotations
# Configurable ordering for memory cards AND context turns — one place, reused by both, so recall and
# management can be prioritized cheaply instead of always scanning in insertion order. Each item is a
# dict; a sort KEY names the field to order by and a direction. Keeping this pure + tiny means the
# expensive part (retrieval) can hand off a pre-scored list and get a stable, cheap ordering back.
from typing import Callable

# name -> (key function, default descending?). "relevance"/"priority"/"size"/"degree" descend by
# default (biggest first); "recency" descends (newest first via the monotonic id); "oldest"/"label"
# ascend. Unknown fields fall back to insertion order.
_KEYS: dict[str, tuple[Callable[[dict], object], bool]] = {
    "relevance": (lambda d: float(d.get("score", 0.0)), True),
    "priority":  (lambda d: (float(d.get("priority", 0)), _seq(d)), True),
    "size":      (lambda d: float(d.get("size", d.get("n_messages", 0))), True),
    "degree":    (lambda d: int(d.get("degree", len(d.get("links", []) or []))), True),
    "recency":   (lambda d: _seq(d), True),
    "oldest":    (lambda d: _seq(d), False),
    "label":     (lambda d: str(d.get("label", "")).lower(), False),
}

SORTS = tuple(_KEYS)


def _seq(d: dict) -> float:
    """A monotonic recency proxy from an 'm-0007'/'c7'-style id (higher = newer). 0 if absent."""
    key = str(d.get("key") or d.get("id") or "")
    digits = "".join(ch for ch in key if ch.isdigit())
    return float(digits) if digits else 0.0


def sort_items(items: list[dict], by: str = "recency", descending: bool | None = None) -> list[dict]:
    """Return items ordered by the named key. `descending=None` uses the key's sensible default
    (relevance/priority/size/degree/recency -> newest/biggest first; oldest/label -> ascending).
    Stable and pure; an unknown `by` leaves the order untouched."""
    spec = _KEYS.get(by)
    if spec is None:
        return list(items)
    keyfn, default_desc = spec
    desc = default_desc if descending is None else descending
    return sorted(items, key=keyfn, reverse=desc)
