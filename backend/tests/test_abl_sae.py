import numpy as np

from crucible.abliteration.sae import SparseAutoencoder


def _sparse_data(n=300, d=12, k_atoms=6, seed=0):
    """Activations are sparse sums of 1-2 unit dictionary atoms — the structure an SAE
    should recover."""
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((k_atoms, d))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    X = np.zeros((n, d))
    for i in range(n):
        chosen = rng.choice(k_atoms, size=rng.integers(1, 3), replace=False)
        for c in chosen:
            X[i] += rng.uniform(0.5, 1.5) * D[c]
    return X, D


def test_sae_learns_and_reconstructs():
    X, _ = _sparse_data()
    sae = SparseAutoencoder(n_features=24, epochs=400, lr=2e-2, l1=5e-3).fit(X)
    # training drives reconstruction error down a lot from the first epoch
    assert sae.history[-1] < 0.3 * sae.history[0]
    assert sae.r2(X) > 0.8                     # explains most variance
    assert sae.reconstruction_error(X) < 0.2


def test_sae_codes_are_sparse():
    X, _ = _sparse_data()
    sae = SparseAutoencoder(n_features=24, epochs=400, lr=2e-2, l1=1e-2).fit(X)
    assert sae.sparsity(X) > 0.5               # most features are off for any given input


def test_decoder_atoms_unit_norm():
    X, _ = _sparse_data()
    sae = SparseAutoencoder(n_features=20, epochs=100).fit(X)
    norms = np.linalg.norm(sae.feature_directions(), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_top_features_returns_active_only():
    X, _ = _sparse_data()
    sae = SparseAutoencoder(n_features=24, epochs=300, lr=2e-2).fit(X)
    feats = sae.top_features(X[0], k=5)
    assert len(feats) >= 1
    assert all(v > 0 for _, v in feats)        # only firing features
    assert feats == sorted(feats, key=lambda t: t[1], reverse=True)


def test_encode_requires_fit():
    try:
        SparseAutoencoder(n_features=4).encode([[0.0, 0.0]])
        assert False
    except RuntimeError:
        pass


def test_determinism():
    X, _ = _sparse_data()
    a = SparseAutoencoder(n_features=16, epochs=120, seed=7).fit(X)
    b = SparseAutoencoder(n_features=16, epochs=120, seed=7).fit(X)
    assert np.allclose(a.feature_directions(), b.feature_directions())


def test_label_features_groups_coherent_tokens():
    from crucible.abliteration.sae import label_features
    # two token groups, each driven by its own axis -> SAE features should separate them
    rng = np.random.default_rng(3)
    d = 8
    X, toks = [], []
    for _ in range(120):
        v = rng.normal(0, 0.02, d); v[0] += 1.0
        X.append(v); toks.append("refuse")
    for _ in range(120):
        v = rng.normal(0, 0.02, d); v[1] += 1.0
        X.append(v); toks.append("hello")
    X = np.array(X)
    sae = SparseAutoencoder(n_features=16, epochs=400, lr=2e-2, l1=5e-3).fit(X)
    labels = label_features(sae, X, toks, n_features=6, n_tokens=6)
    assert len(labels) >= 2
    # at least one feature fires predominantly on a single coherent token
    assert any(len(set(l["fires_on"])) == 1 for l in labels)
    assert all(l["peak"] > 0 for l in labels)
