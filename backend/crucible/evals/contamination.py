from __future__ import annotations
# Benchmark contamination check. A measured score is only trustworthy if the eval items
# weren't memorized. This flags overlap between a candidate text (a model output, or a
# training shard) and a reference (the benchmark item) by n-gram containment — the standard
# cheap contamination signal. Pure; no model needed.
import re

_WORD = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = _tokens(text)
    if len(toks) < n:
        return set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def contamination_score(candidate: str, reference: str, n: int = 13) -> float:
    """Fraction of the reference's n-grams that also appear in the candidate.
    1.0 = the reference is reproduced verbatim (strong contamination signal); 0.0 = none."""
    ref = ngrams(reference, n)
    if not ref:
        # reference shorter than n tokens: fall back to a smaller window
        ref = ngrams(reference, max(1, len(_tokens(reference))))
        if not ref:
            return 0.0
    cand = ngrams(candidate, n) | ngrams(candidate, max(1, len(_tokens(reference)))) \
        if len(_tokens(candidate)) < n else ngrams(candidate, n)
    overlap = len(ref & cand)
    return overlap / len(ref)


def flag_contamination(candidate: str, reference: str, n: int = 13,
                       threshold: float = 0.5) -> dict:
    score = contamination_score(candidate, reference, n)
    return {"score": score, "n": n, "threshold": threshold, "contaminated": score >= threshold}
