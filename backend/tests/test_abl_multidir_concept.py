import numpy as np

from crucible.abliteration.multidir import refusal_directions, sticky_fraction
from crucible.abliteration.concept import concept_vector, project_strength, separability


def _hetero_data(n=200, seed=0):
    """Heterogeneous refusal: the first half of harmful prompts refuse along axis 0, the
    second half along axis 1. The mean difference alone is one blended vector — only the
    per-example difference SVD recovers BOTH axes."""
    rng = np.random.default_rng(seed)
    dim = 5
    harmless = rng.normal(0, 0.03, size=(n, dim))
    harmful = harmless.copy()
    half = n // 2
    harmful[:half, 0] += 1.0     # stronger axis -> larger singular value
    harmful[half:, 1] += 0.7     # weaker axis
    return harmful, harmless


def test_multiple_directions_finds_both_axes():
    harmful, harmless = _hetero_data()
    dirs, seps = refusal_directions(harmful, harmless, k=3)
    assert dirs.shape[0] >= 2
    # the two leading directions align with axis 0 and axis 1 (in some order)
    leading_axes = {int(np.argmax(np.abs(dirs[0]))), int(np.argmax(np.abs(dirs[1])))}
    assert leading_axes == {0, 1}
    # orthonormal
    assert abs(float(dirs[0] @ dirs[1])) < 1e-6
    assert abs(float(np.linalg.norm(dirs[0])) - 1.0) < 1e-9
    # stronger axis carries the larger singular value
    assert seps[0] > seps[1]


def test_sticky_fraction_flags_non_rank1():
    _, seps = refusal_directions(*_hetero_data(), k=3)
    assert sticky_fraction(seps) > 0.2           # substantial refusal beyond the primary axis
    assert sticky_fraction([1.0]) == 0.0         # perfectly rank-1
    assert sticky_fraction([]) == 0.0


def test_multidir_empty_when_no_separation():
    rng = np.random.default_rng(1)
    same = rng.normal(0, 0.1, size=(50, 4))
    dirs, seps = refusal_directions(same, same.copy(), k=4)
    assert dirs.shape[0] == 0 and seps == []


def test_concept_vector_and_strength():
    pos = np.array([[2.0, 0.0], [2.0, 0.0]])
    neg = np.array([[0.0, 0.0], [0.0, 0.0]])
    v = concept_vector(pos, neg)
    assert np.allclose(v, [2.0, 0.0])
    u = concept_vector(pos, neg, normalize=True)
    assert np.allclose(np.linalg.norm(u), 1.0)
    # an activation expressing the concept projects positively; its opposite negatively
    assert project_strength([3.0, 0.0], v) > 0
    assert project_strength([-3.0, 0.0], v) < 0


def test_concept_separability_high_when_linearly_encoded():
    rng = np.random.default_rng(2)
    pos = rng.normal(0, 0.05, (100, 3)); pos[:, 0] += 1.0
    neg = rng.normal(0, 0.05, (100, 3))
    v = concept_vector(pos, neg)
    assert separability(pos, neg, v) > 3.0       # cleanly separated


def test_concept_vector_rejects_empty_side():
    try:
        concept_vector(np.zeros((0, 3)), np.ones((2, 3)))
        assert False, "expected ValueError"
    except ValueError:
        pass
