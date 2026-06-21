# Logit-lens feature decoder: translate an abstract refusal DIRECTION into the actual
# WORDS it promotes/suppresses, by projecting it through the unembedding. This is what
# turns "layer 23, margin 13.93" into "this feature makes the model say sorry / cannot
# / apologize and stops it saying Sure / Here / Step" — human-readable censorship.
import numpy as np
from numpy.typing import ArrayLike


def decode_direction(unembed: ArrayLike, direction: ArrayLike, decode, top_k: int = 15) -> dict:
    u = np.asarray(unembed, dtype=np.float64)          # (vocab, hidden)
    d = np.asarray(direction, dtype=np.float64)        # (hidden,)
    scores = u @ d                                      # (vocab,)
    order = np.argsort(scores)
    top = order[::-1][:top_k]
    bottom = order[:top_k]
    promoted = [{"token": decode(int(i)), "score": float(scores[i])} for i in top]
    suppressed = [{"token": decode(int(i)), "score": float(scores[i])} for i in bottom]
    return {"promoted": promoted, "suppressed": suppressed}
