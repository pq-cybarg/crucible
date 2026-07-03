import numpy as np

from crucible.abliteration.tuned_lens import TunedLens, fit_affine


def test_fit_affine_recovers_linear_map():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    A_true = rng.standard_normal((5, 3))
    b_true = rng.standard_normal(3)
    Y = X @ A_true + b_true
    A, b = fit_affine(X, Y)
    assert np.allclose(A, A_true, atol=1e-2)
    assert np.allclose(b, b_true, atol=1e-2)


def test_tuned_lens_beats_raw_for_rotated_layer():
    rng = np.random.default_rng(1)
    n, d = 300, 6
    final = rng.standard_normal((n, d))
    # an early layer is the final state in a DIFFERENT basis (rotated) -> raw lens fails,
    # tuned lens (which learns the rotation) succeeds
    R = np.linalg.qr(rng.standard_normal((d, d)))[0]
    early = final @ R
    lens = TunedLens().fit({4: early}, final)
    assert lens.residual(4, early, final) < 0.1 * lens.raw_residual(early, final)
    assert lens.decodability(4, early, final) > 0.95


def test_decodability_curve_rises_toward_final():
    rng = np.random.default_rng(2)
    n, d = 250, 5
    final = rng.standard_normal((n, d))
    # layer 0 = noise, layer 8 = final + small noise -> decodability should climb
    layers = {0: rng.standard_normal((n, d)),
              8: final + rng.normal(0, 0.05, (n, d))}
    lens = TunedLens().fit(layers, final)
    curve = lens.curve(layers, final)
    by_layer = {r["layer"]: r["decodability"] for r in curve}
    assert by_layer[8] > by_layer[0]
    assert by_layer[8] > 0.9


def test_curve_reports_both_residuals():
    rng = np.random.default_rng(3)
    final = rng.standard_normal((100, 4))
    lens = TunedLens().fit({2: final.copy()}, final)
    row = lens.curve({2: final.copy()}, final)[0]
    assert "tuned_residual" in row and "raw_residual" in row and row["layer"] == 2
