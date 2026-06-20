import numpy as np

from crucible.abliteration.orthogonalize import (
    orthogonalize_embedding, orthogonalize_writing_matrix, project_out)


def unit(v):
    return np.asarray(v, dtype=float) / np.linalg.norm(v)


def test_project_out_vector_is_orthogonal():
    r = unit([1.0, 1.0, 0.0])
    out = project_out(np.array([3.0, 1.0, 2.0]), r)
    assert np.isclose(out @ r, 0.0)


def test_project_out_matrix_rows_orthogonal():
    r = unit([0.0, 1.0, 0.0])
    out = project_out(np.random.default_rng(0).standard_normal((5, 3)), r)
    assert np.allclose(out @ r, 0.0, atol=1e-9)


def test_writing_matrix_has_no_refusal_component():
    rng = np.random.default_rng(1)
    r = unit([1.0, 0.0, 0.0, 0.0])
    W = rng.standard_normal((4, 7))
    Wp = orthogonalize_writing_matrix(W, r)
    assert np.allclose(r @ Wp, 0.0, atol=1e-9)


def test_embedding_rows_have_no_refusal_component():
    rng = np.random.default_rng(2)
    r = unit([0.0, 0.0, 1.0, 0.0])
    E = rng.standard_normal((10, 4))
    Ep = orthogonalize_embedding(E, r)
    assert np.allclose(Ep @ r, 0.0, atol=1e-9)
