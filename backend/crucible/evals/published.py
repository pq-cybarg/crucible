from __future__ import annotations
# Cited public numbers. GLM-5 family figures are from public reporting (2026);
# Opus entries are intentionally None - fill with an official Anthropic citation
# rather than guessing. Never present an uncited number as measured.
_GLM_SRC = "https://medium.com/@mlabonne/glm-5-chinas-first-public-ai-company-ships-a-frontier-model-a068cecb74e3"

PUBLISHED: dict[str, dict[str, dict]] = {
    "GLM-5.2 family": {
        "SWE-bench Verified": {"value": 0.778, "source": _GLM_SRC},
        "AIME 2026": {"value": 0.927, "source": _GLM_SRC},
        "GPQA-Diamond": {"value": 0.860, "source": _GLM_SRC},
    },
    "Claude Opus 4.x": {
        "SWE-bench Verified": {"value": None, "source": "cite from Anthropic"},
        "AIME 2026": {"value": None, "source": "cite from Anthropic"},
        "GPQA-Diamond": {"value": None, "source": "cite from Anthropic"},
    },
}
