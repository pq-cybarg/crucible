# Logit-lens feature decoder: translate an abstract refusal DIRECTION into the actual
# WORDS it promotes/suppresses, by projecting it through the unembedding. This is what
# turns "layer 23, margin 13.93" into "this feature makes the model say sorry / cannot
# / apologize and stops it saying Sure / Here / Step" — human-readable censorship.
import numpy as np
from numpy.typing import ArrayLike


def _word_like(tok: str) -> bool:
    t = tok.strip()
    return 0 < len(t) <= 24 and any(c.isalpha() for c in t)


def decode_direction(unembed: ArrayLike, direction: ArrayLike, decode, top_k: int = 15,
                     keep=_word_like) -> dict:
    u = np.asarray(unembed, dtype=np.float64)          # (vocab, hidden)
    d = np.asarray(direction, dtype=np.float64)        # (hidden,)
    scores = u @ d                                      # (vocab,)
    order = np.argsort(scores)

    def collect(indices):
        out = []
        for i in indices:
            tok = decode(int(i))
            if keep is None or keep(tok):
                out.append({"token": tok, "score": float(scores[i])})
                if len(out) >= top_k:
                    break
        return out

    return {"promoted": collect(order[::-1]), "suppressed": collect(order)}
