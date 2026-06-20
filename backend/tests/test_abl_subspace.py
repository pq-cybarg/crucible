import numpy as np

from crucible.abliteration.subspace import refusal_subspace


def test_rank1_recovers_direction():
    e = np.zeros(6); e[2] = 1.0
    harmful = np.random.default_rng(0).standard_normal((8, 6)) * 0.05 + 4 * e
    harmless = np.random.default_rng(1).standard_normal((8, 6)) * 0.05
    dirs, ev = refusal_subspace(harmful, harmless, k=1)
    assert dirs.shape == (1, 6)
    assert abs(abs(np.dot(dirs[0], e)) - 1.0) < 0.05
    assert ev == [1.0]


def test_rank_k_orthonormal():
    rng = np.random.default_rng(2)
    harmful = rng.standard_normal((10, 8)) + 3 * np.eye(8)[1]
    harmless = rng.standard_normal((10, 8))
    dirs, ev = refusal_subspace(harmful, harmless, k=3)
    assert dirs.shape == (3, 8)
    # rows orthonormal
    gram = dirs @ dirs.T
    assert np.allclose(gram, np.eye(3), atol=1e-6)
    assert len(ev) == 3 and all(0.0 <= x <= 1.0001 for x in ev)
