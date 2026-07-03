from __future__ import annotations
# Provider routing. Crucible can act as an OpenAI-compatible model PROVIDER: OpenCode (or any
# client) points at Crucible's /v1, and Crucible decides which backing model actually serves
# each request. The policy: honor the requested model if it's available; otherwise fall back
# to a preconfigured preference order; otherwise pick the nearest available model — where
# "available" is tested at request time (a live health check), not assumed. Pure + tested;
# the availability probe is injected.
from typing import Callable, Optional

# ids the client may send to mean "you pick" rather than a specific model.
AUTO_ALIASES = {"", "auto", "crucible", "nearest", "default", "crucible-local", "local"}


def choose_model(requested: Optional[str], candidates: list[str],
                 preferences: Optional[list[str]] = None,
                 is_available: Optional[Callable[[str], bool]] = None) -> Optional[str]:
    """Return the model id to serve this request, or None if nothing is available.

    Order: explicit request (if a real, available model) -> preference order -> nearest
    available candidate. `is_available(id)` is called at request time (defaults to all-up)."""
    avail = is_available or (lambda _id: True)
    present = set(candidates)

    # 1. explicit, specific request that exists and is up
    if requested and requested not in AUTO_ALIASES and requested in present and avail(requested):
        return requested

    # 2. preconfigured preference order
    for pid in (preferences or []):
        if pid in present and avail(pid):
            return pid

    # 3. nearest available — first candidate that answers
    for pid in candidates:
        if avail(pid):
            return pid
    return None


def routing_explain(requested: Optional[str], chosen: Optional[str],
                    preferences: Optional[list[str]]) -> str:
    """One-line reason a model was chosen (for the response's system_fingerprint/logs)."""
    if chosen is None:
        return "no model available"
    if requested and requested not in AUTO_ALIASES and requested == chosen:
        return f"served requested model '{chosen}'"
    if preferences and chosen in preferences:
        return f"requested unavailable -> preference '{chosen}'"
    return f"requested unavailable -> nearest available '{chosen}'"
