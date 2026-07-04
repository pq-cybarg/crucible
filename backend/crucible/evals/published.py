from __future__ import annotations
# Cited public numbers for head-to-head CONTEXT — never presented as our own measurement, and
# never as authoritative. Each figure carries its provenance: `verified` (is the source a primary,
# official publication?) and `source_type` ("secondary" reporting vs "official" vendor/paper vs
# "unset"). The GLM-5 figures come from SECONDARY reporting (a blog aggregating the release), so
# they're flagged verified=False — treat as indicative, not gospel. Opus entries are intentionally
# blank (value=None) to be filled from an OFFICIAL Anthropic citation rather than guessed. To get
# a number YOU can stand behind, run the model through the real lm-eval harness (/api/evals/lmeval).

_GLM_SRC = "https://medium.com/@mlabonne/glm-5-chinas-first-public-ai-company-ships-a-frontier-model-a068cecb74e3"


def _secondary(value: float) -> dict:
    return {"value": value, "source": _GLM_SRC, "source_type": "secondary",
            "verified": False, "note": "secondary reporting — not an official source; verify before citing"}


def _unset() -> dict:
    return {"value": None, "source": "cite from Anthropic (official)", "source_type": "unset",
            "verified": False, "note": "fill from an official Anthropic publication; do not guess"}


PROVIDERS: dict[str, dict[str, dict]] = {
    "GLM-5.2 family": {
        "SWE-bench Verified": _secondary(0.778),
        "AIME 2026": _secondary(0.927),
        "GPQA-Diamond": _secondary(0.860),
    },
    "Claude Opus 4.x": {
        "SWE-bench Verified": _unset(),
        "AIME 2026": _unset(),
        "GPQA-Diamond": _unset(),
    },
}

# Backward-compatible alias — callers that want the raw provider->benchmark->cell table.
PUBLISHED = PROVIDERS

# What the API returns: the table plus an explicit honesty banner so the GUI can badge unverified
# figures rather than rendering a blog number as if it were authoritative.
PUBLISHED_PAYLOAD: dict = {
    "providers": PROVIDERS,
    "disclaimer": ("Context only — figures are cited from their sources, not measured here. "
                   "Rows marked verified=false come from secondary reporting or are unset; run "
                   "lm-eval for a number you can stand behind."),
}
