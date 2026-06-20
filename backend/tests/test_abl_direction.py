import numpy as np
import pytest

from crucible.abliteration.direction import compute_refusal_direction


def test_direction_points_from_harmless_to_harmful():
    harmful = np.array([[1.0, 0.0], [1.0, 0.0]])
    harmless = np.array([[0.0, 0.0], [0.0, 0.0]])
    d = compute_refusal_direction(harmful, harmless)
    assert np.allclose(d, [1.0, 0.0])
    assert np.isclose(np.linalg.norm(d), 1.0)


def test_zero_difference_raises():
    x = np.ones((3, 4))
    with pytest.raises(ValueError):
        compute_refusal_direction(x, x)
