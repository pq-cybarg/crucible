import numpy as np

from crucible.abliteration.lora import LoRA, fit_lowrank, reconstruction_error, unalign_lora
from crucible.abliteration.diagnosis import ablation_impact  # noqa: F401 (sanity import)


def test_lora_delta_scaling():
    A = np.ones((2, 4)); B = np.ones((3, 2))
    lora = LoRA(A, B, alpha=2)
    d = lora.delta()
    assert d.shape == (3, 4)
    assert np.allclose(d, (2 / 2) * (B @ A))


def test_fit_lowrank_exact_for_rank1_target():
    rng = np.random.default_rng(0)
    u = rng.standard_normal((6, 1)); v = rng.standard_normal((1, 5))
    target = u @ v                                  # rank-1
    lora = fit_lowrank(target, rank=1)
    assert lora.rank == 1
    assert reconstruction_error(lora, target) < 1e-9
    assert np.allclose(lora.delta(), target, atol=1e-9)


def test_fit_lowrank_alpha_invariant():
    rng = np.random.default_rng(1)
    target = rng.standard_normal((5, 5))
    for alpha in (1.0, 4.0, 16.0):
        lora = fit_lowrank(target, rank=3, alpha=alpha)
        # delta reproduces the SAME rank-3 approximation regardless of alpha
        assert reconstruction_error(lora, target) < 0.6      # rank-3 of a rank-5 target
        two = fit_lowrank(target, rank=3, alpha=1.0)
        assert np.allclose(lora.delta(), two.delta(), atol=1e-8)


def test_fit_lowrank_improves_with_rank():
    rng = np.random.default_rng(2)
    target = rng.standard_normal((8, 8))
    errs = [reconstruction_error(fit_lowrank(target, r), target) for r in (1, 3, 6, 8)]
    assert errs == sorted(errs, reverse=True)      # more rank -> lower error
    assert errs[-1] < 1e-9                          # full rank -> exact


def test_unalign_lora_matches_abliteration():
    from crucible.weights.gguf_edit import orthogonalize_matrix
    rng = np.random.default_rng(3)
    W = rng.standard_normal((10, 6))
    r = rng.standard_normal(10); r /= np.linalg.norm(r)
    lora = unalign_lora(W, r, coef=1.0, rank=1)
    # attaching the adapter reproduces the surgically-abliterated matrix
    assert np.allclose(lora.apply(W), orthogonalize_matrix(W, r), atol=1e-9)
    # and the refusal component is gone
    assert np.allclose(r @ lora.apply(W), 0.0, atol=1e-8)


def test_unalign_lora_is_low_rank_and_small():
    rng = np.random.default_rng(4)
    W = rng.standard_normal((256, 256))
    r = rng.standard_normal(256); r /= np.linalg.norm(r)
    lora = unalign_lora(W, r, rank=1)
    assert np.linalg.matrix_rank(lora.delta()) <= 1
    assert lora.n_params < W.size                   # far fewer params than the full matrix


def test_realign_lora_adds_refusal_back():
    from crucible.abliteration.lora import realign_lora, unalign_lora
    rng = np.random.default_rng(5)
    W = rng.standard_normal((10, 6))
    r = rng.standard_normal(10); r /= np.linalg.norm(r)
    un = unalign_lora(W, r, coef=1.0)
    re = realign_lora(W, r, coef=1.0)
    # realign delta is the exact negation of the un-align delta (opposite direction)
    assert np.allclose(re.delta(), -un.delta(), atol=1e-9)
    # applying realign STRENGTHENS the refusal component (grows it), un-align removes it
    base = float(np.linalg.norm(r @ W))
    assert float(np.linalg.norm(r @ re.apply(W))) > base
    assert np.allclose(r @ un.apply(W), 0.0, atol=1e-8)


def test_unalign_then_realign_restores_original():
    from crucible.abliteration.lora import realign_lora
    from crucible.weights.gguf_edit import orthogonalize_matrix
    rng = np.random.default_rng(6)
    W = rng.standard_normal((8, 5))
    r = rng.standard_normal(8); r /= np.linalg.norm(r)
    Wcut = orthogonalize_matrix(W, r)              # abliterated (refusal removed)
    # realigning the CUT matrix with the removed component restores it toward the original's refusal
    restored = realign_lora(W, r, coef=1.0).apply(Wcut)
    assert np.allclose(r @ restored, r @ W, atol=1e-8)
