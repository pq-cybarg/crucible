from __future__ import annotations
# Behavioral before/after verification: does the abliterated model comply with what
# the base refused (uncensoring worked), without over-refusing benign prompts
# (surgical)? Solver-agnostic so it unit-tests with fakes and runs on torch adapters.
from typing import Callable

from crucible.abliteration.detection import refusal_rate

Solver = Callable[[str], str]


def behavioral_compare(base_solver: Solver, variant_solver: Solver,
                       harmful: list[str], benign: list[str]) -> dict:
    base_harmful = [base_solver(p) for p in harmful]
    var_harmful = [variant_solver(p) for p in harmful]
    base_benign = [base_solver(p) for p in benign]
    var_benign = [variant_solver(p) for p in benign]

    base_hr = refusal_rate(base_harmful)
    var_hr = refusal_rate(var_harmful)
    return {
        "harmful_refusal_rate": {"before": base_hr, "after": var_hr},
        "harmful_compliance_rate": {"before": 1.0 - base_hr, "after": 1.0 - var_hr},
        "benign_over_refusal_rate": {
            "before": refusal_rate(base_benign), "after": refusal_rate(var_benign)},
        "samples": [
            {"prompt": p, "before": base_harmful[i], "after": var_harmful[i]}
            for i, p in enumerate(harmful)
        ],
    }
