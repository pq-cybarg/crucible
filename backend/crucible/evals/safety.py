from __future__ import annotations
from typing import Callable

from crucible.abliteration.detection import refusal_rate

Solver = Callable[[str], str]


def safety_metrics(solver: Solver, harmful: list[str], benign: list[str]) -> dict:
    harmful_answers = [solver(p) for p in harmful]
    benign_answers = [solver(p) for p in benign]
    h_refusal = refusal_rate(harmful_answers)
    return {
        "harmful_refusal_rate": h_refusal,
        "harmful_compliance_rate": 1.0 - h_refusal,
        "benign_over_refusal_rate": refusal_rate(benign_answers),
    }
