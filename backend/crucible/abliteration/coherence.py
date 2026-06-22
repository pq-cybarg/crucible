# Coherence proxy for the insertion auto-tuner: penalize gibberish (non-word chars) and
# repetition, so the search can avoid additive doses that wreck the model's output.
def coherence_score(text: str) -> float:
    t = text.strip()
    if not t:
        return 0.0
    ok = sum(1 for c in t if c.isalnum() or c.isspace() or c in ".,!?';:-()")
    alpha_ratio = ok / len(t)
    toks = t.split()
    unique_ratio = len(set(toks)) / max(1, len(toks))
    return round(alpha_ratio * unique_ratio, 4)
