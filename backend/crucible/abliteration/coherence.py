# Coherence proxy for the insertion auto-tuner: penalize gibberish (non-ASCII chars,
# no spaces / concatenated runs, repetition) so the search avoids doses that wreck output.
def coherence_score(text: str) -> float:
    t = text.strip()
    if not t:
        return 0.0
    good = sum(1 for c in t if c.isascii() and (c.isalnum() or c.isspace() or c in ".,!?';:-()"))
    ascii_ratio = good / len(t)
    toks = t.split()
    unique_ratio = len(set(toks)) / max(1, len(toks))
    # real prose has roughly one space per ~6 chars; concatenated gibberish has none
    space_density = min(1.0, t.count(" ") / max(1.0, len(t) / 6.0))
    return round(ascii_ratio * unique_ratio * space_density, 4)
