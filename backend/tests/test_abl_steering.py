import numpy as np

from crucible.abliteration.steering import steer


def test_steer_adds_scaled_vector():
    x = np.array([1.0, 2.0])
    out = steer(x, np.array([1.0, 0.0]), 3.0)
    assert np.allclose(out, [4.0, 2.0])


def test_steer_is_reversible():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 6))
    v = rng.standard_normal(6)
    back = steer(steer(x, v, 2.5), v, -2.5)
    assert np.allclose(back, x)
