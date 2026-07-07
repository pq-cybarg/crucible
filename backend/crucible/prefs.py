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

_MODES = ("allow", "ask", "deny")


def _default_permissions() -> dict:
    # Persisted tool-permission DEFAULTS the forge applies to every run: a global default mode,
    # per-tool overrides, and path-scoped rules (limited-permission directories/files).
    return {"default": "ask", "modes": {}, "path_rules": []}


def _default_resource_limits() -> dict:
    # Memory/compute caps applied to Ollama models via its NATIVE /api/chat (the OpenAI-compat endpoint
    # ignores these). Trading RAM for time so big local models stop freezing the machine:
    #   num_ctx           - context window = KV-cache size. The big RAM lever. 0 = model default.
    #   keep_alive        - how long Ollama keeps the model resident after a reply ("0" = unload now and
    #                       free the weights between turns; "5m", "-1" = forever). "" = default (5m).
    #   max_output_tokens - cap generation length. 0 = uncapped.
    #   num_gpu           - layers offloaded to GPU/Metal; lower keeps more on CPU (slower, less VRAM).
    #                       -1 = auto.
    return {"num_ctx": 0, "keep_alive": "", "max_output_tokens": 0, "num_gpu": -1}


DEFAULTS: dict = {
    "default_sort": "recency",
    "balanced_recency_weight": 0.5,
    "default_metric": "bm25",
    "processing_model": None,
    "permissions": _default_permissions(),
    "resource_limits": _default_resource_limits(),
}


def _clean_permissions(data: dict) -> dict:
    out = _default_permissions()
    if isinstance(data, dict):
        if data.get("default") in _MODES:
            out["default"] = data["default"]
        modes = data.get("modes")
        if isinstance(modes, dict):
            out["modes"] = {str(k): v for k, v in modes.items() if v in _MODES}
        rules = data.get("path_rules")
        if isinstance(rules, list):
            clean = []
            for r in rules:
                if isinstance(r, dict) and str(r.get("glob", "")).strip() and r.get("mode", "deny") in _MODES:
                    tools = r.get("tools") or []
                    clean.append({"glob": str(r["glob"]).strip(), "mode": r.get("mode", "deny"),
                                  "tools": [str(t) for t in tools if str(t).strip()]})
            out["path_rules"] = clean
    return out


def _clean_resource_limits(data: dict) -> dict:
    out = _default_resource_limits()
    if not isinstance(data, dict):
        return out

    def _int(key: str, lo: int) -> None:
        try:
            out[key] = max(lo, int(data.get(key, out[key])))
        except (TypeError, ValueError):
            pass

    _int("num_ctx", 0)
    _int("max_output_tokens", 0)
    _int("num_gpu", -1)
    ka = data.get("keep_alive", "")
    out["keep_alive"] = str(ka).strip() if ka is not None else ""
    return out


def has_limits(rl: dict) -> bool:
    """True if any resource limit is set away from its default — i.e. Crucible should route Ollama
    through its native /api/chat to honor them instead of the (limit-ignoring) OpenAI endpoint."""
    d = _default_resource_limits()
    return bool(rl) and any(rl.get(k) != d[k] for k in d)


def _clean(data: dict) -> dict:
    """Coerce/validate a raw dict onto DEFAULTS — unknown sorts/metrics/modes and out-of-range weights
    fall back rather than corrupting downstream ordering or leaking a permission."""
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
    out["permissions"] = _clean_permissions(data.get("permissions", {}))
    out["resource_limits"] = _clean_resource_limits(data.get("resource_limits", {}))
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
