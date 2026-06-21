import numpy as np

from crucible.abliteration.lens import decode_direction


def test_decode_promotes_aligned_token():
    # vocab of 5 tokens, hidden 3; token 2's row aligns with the direction
    unembed = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
    direction = np.array([0.0, 1.0, 0.0])
    out = decode_direction(unembed, direction, decode=lambda i: f"tok{i}", top_k=2)
    assert out["promoted"][0]["token"] == "tok2"       # highest dot with [0,1,0]
    assert out["suppressed"][0]["token"] == "tok4"     # most negative
