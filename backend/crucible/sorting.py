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

# "balanced" is not a per-item key — it needs list-wide min/max to normalize — so it lives outside
# _KEYS and is handled specially in sort_items. It blends recency and priority (salience) the way human
# recall does: privilege what's *recent* AND what *matters*, instead of pure positional (recency/oldest)
# bias. See _balanced_scores.
SORTS = (*_KEYS, "balanced")

# Default weight on recency vs. priority for the balanced sort. 0.5 = equal; ->1 leans recency,
# ->0 leans salience. Exposed as a parameter so a caller/preference can tune the human-recall curve.
BALANCED_RECENCY_WEIGHT = 0.5


def _seq(d: dict) -> float:
    """A monotonic recency proxy from an 'm-0007'/'c7'-style id (higher = newer). 0 if absent."""
    key = str(d.get("key") or d.get("id") or "")
    digits = "".join(ch for ch in key if ch.isdigit())
    return float(digits) if digits else 0.0


def _norm(values: list[float]) -> dict[int, float]:
    """Min-max normalize to [0,1] by index. A flat set (all equal) maps to 0.5 so it contributes
    neutrally rather than dominating."""
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:
        return {i: 0.5 for i in range(len(values))}
    return {i: (v - lo) / span for i, v in enumerate(values)}


def _balanced_scores(items: list[dict], recency_weight: float) -> list[float]:
    """Salience-weighted recency: score = w·norm(recency) + (1−w)·norm(priority). Normalizing each
    axis across the set keeps the two comparable regardless of their raw scales (ids climb into the
    thousands; priorities are small integers)."""
    w = max(0.0, min(1.0, recency_weight))
    rec = _norm([_seq(d) for d in items])
    pri = _norm([float(d.get("priority", 0)) for d in items])
    return [w * rec[i] + (1.0 - w) * pri[i] for i in range(len(items))]


def sort_items(
    items: list[dict],
    by: str = "recency",
    descending: bool | None = None,
    recency_weight: float = BALANCED_RECENCY_WEIGHT,
) -> list[dict]:
    """Return items ordered by the named key. `descending=None` uses the key's sensible default
    (relevance/priority/size/degree/recency -> newest/biggest first; oldest/label -> ascending).
    `balanced` blends recency + priority (salience) with `recency_weight` in [0,1]. Stable and pure;
    an unknown `by` leaves the order untouched."""
    if by == "balanced":
        scores = _balanced_scores(items, recency_weight)
        order = sorted(range(len(items)), key=lambda i: scores[i], reverse=descending is not False)
        return [items[i] for i in order]
    spec = _KEYS.get(by)
    if spec is None:
        return list(items)
    keyfn, default_desc = spec
    desc = default_desc if descending is None else descending
    return sorted(items, key=keyfn, reverse=desc)
