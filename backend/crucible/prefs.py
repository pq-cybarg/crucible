from __future__ import annotations
# Organizational preferences for memory & context: how recall is ordered and how "closeness" is
# measured by default. Kept tiny and validated — an out-of-range value falls back to the safe default
# rather than corrupting a sort. Persisted as JSON like the other stores.
#
#   default_sort            - the ordering used when a caller doesn't specify one. Primacy (oldest),
#                             recency, salience (priority), or the human-recall blend (balanced).
#   balanced_recency_weight - w in [0,1] for the balanced sort: ->1 leans recency, ->0 leans salience.
#   default_metric          - which distance/similarity family search uses by default.
#   processing_model        - the (preferably small, cheap) model id used for llm-judged distance and
#                             background reorganization — the brain-plasticity / preprocessing role.
#                             None means llm-judged metrics are unavailable until one is chosen.
import json
from pathlib import Path

from crucible.metrics import METRICS
from crucible.sorting import SORTS

DEFAULTS: dict = {
    "default_sort": "recency",
    "balanced_recency_weight": 0.5,
    "default_metric": "bm25",
    "processing_model": None,
}


def _clean(data: dict) -> dict:
    """Coerce/validate a raw dict onto DEFAULTS — unknown sorts/metrics and out-of-range weights fall
    back rather than corrupting downstream ordering."""
    out = dict(DEFAULTS)
    if data.get("default_sort") in SORTS:
        out["default_sort"] = data["default_sort"]
    if data.get("default_metric") in METRICS:
        out["default_metric"] = data["default_metric"]
    try:
        w = float(data.get("balanced_recency_weight", DEFAULTS["balanced_recency_weight"]))
        out["balanced_recency_weight"] = max(0.0, min(1.0, w))
    except (TypeError, ValueError):
        pass
    pm = data.get("processing_model")
    out["processing_model"] = str(pm) if pm else None
    return out


class PreferencesStore:
    """Persists organizational preferences to a JSON file (like the profile/recipe stores)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def get(self) -> dict:
        try:
            return _clean(json.loads(self.path.read_text()))
        except (FileNotFoundError, json.JSONDecodeError):
            return dict(DEFAULTS)

    def save(self, data: dict) -> dict:
        merged = _clean({**self.get(), **(data or {})})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(merged, indent=2))
        return merged
