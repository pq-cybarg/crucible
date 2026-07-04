from __future__ import annotations
# Degeneration guard for the insertion / abliteration auto-tuner. This is NOT a semantic
# coherence or correctness measure — it only flags BROKEN output (repetition loops,
# concatenated-run gibberish, control-char/replacement-char corruption) so the search avoids
# doses that wreck generation. It is language-agnostic: it does NOT penalize non-ASCII scripts
# (CJK/Arabic/Cyrillic are perfectly valid output). Named honestly; `coherence_score` is kept
# as a backward-compatible alias.
import re

_GARBAGE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")   # control chars + U+FFFD replacement
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")   # scripts that don't use spaces


def degeneration_guard(text: str) -> float:
    """0..1 — how INTACT the output looks (1 = clean generation, 0 = degenerate). Combines:
    non-repetition (unique-token ratio), freedom from garbage/control chars, and word/segment
    structure. Language-agnostic. Only catches broken output — not whether it's true or good."""
    t = (text or "").strip()
    if not t:
        return 0.0
    toks = t.split()
    non_repetition = len(set(toks)) / max(1, len(toks))
    garbage = len(_GARBAGE.findall(t))
    clean = 1.0 - garbage / len(t)
    # structured = has word breaks (spaces) OR is a spaceless script (CJK). Concatenated
    # latin gibberish ("thisisonelongrun...") has neither and is penalized.
    structured = 1.0 if (" " in t or _CJK.search(t)) else 0.4
    return round(non_repetition * clean * structured, 4)


# Backward-compatible alias — callers said "coherence" but it always meant this guard.
coherence_score = degeneration_guard
