from __future__ import annotations
import re

import numpy as np


def extract_choice(text: str) -> str | None:
    m = re.search(r"\b([A-E])\b", text.upper())
    return m.group(1) if m else None


def mc_accuracy(predictions: list[str], answers: list[str]) -> float:
    if not answers:
        return 0.0
    hits = sum(1 for p, a in zip(predictions, answers) if p == a)
    return hits / len(answers)


def expected_calibration_error(confidences: list[float], correct: list[bool],
                               n_bins: int = 10) -> float:
    conf = np.asarray(confidences, dtype=np.float64)
    cor = np.asarray(correct, dtype=np.float64)
    n = len(conf)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / n) * abs(float(cor[mask].mean()) - float(conf[mask].mean()))
    return float(ece)
