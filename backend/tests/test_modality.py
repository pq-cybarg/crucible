"""Modality refusal/safety directions: contrastive diff-of-means in an encoder's embedding space,
scored by HELD-OUT (cross-validated) separability so the number is honest — ~0 for unrelated data,
never the in-sample optimism that makes random clusters look separated when dim > n."""
import numpy as np
import pytest

from crucible.abliteration.modality import (MODALITIES, held_out_separability, modality_direction,
                                            summarize_modality)


def _clusters(n, dim, shift_dim, shift, seed):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, dim))
    s = np.zeros(dim); s[shift_dim] = shift
    return base + s, rng.standard_normal((n, dim))


def test_modality_direction_is_unit_and_points_harmful():
    harmful, benign = _clusters(20, 16, 3, 5.0, 0)
    d = modality_direction(harmful, benign)
    assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-6
    # projecting the (harmful - benign) mean gap onto d is positive
    gap = harmful.mean(0) - benign.mean(0)
    assert float(gap @ d) > 0


def test_held_out_is_high_for_real_separation():
    harmful, benign = _clusters(20, 16, 3, 5.0, 1)
    assert held_out_separability(harmful, benign) > 1.0


def test_held_out_is_near_zero_for_random_on_average():
    # in-sample would be spuriously large; held-out is unbiased (~0) averaged over seeds
    vals = []
    for s in range(20):
        rng = np.random.default_rng(s)
        vals.append(held_out_separability(rng.standard_normal((40, 16)), rng.standard_normal((40, 16))))
    assert abs(float(np.mean(vals))) < 0.3


def test_summarize_flags_real_as_encoded_reliable():
    harmful, benign = _clusters(20, 16, 3, 5.0, 2)
    out = summarize_modality(harmful, benign, "image")
    assert out["modality"] == "image" and out["dim"] == 16
    assert out["reliable"] is True and out["linearly_encoded"] is True
    assert "held-out" in out["separability_kind"]
    assert out["separability"] > 1.0 and "in_sample_separability" in out


def test_summarize_does_not_claim_encoding_on_random_large_n():
    rng = np.random.default_rng(7)
    out = summarize_modality(rng.standard_normal((40, 16)), rng.standard_normal((40, 16)), "audio")
    assert out["reliable"] is True and out["linearly_encoded"] is False


def test_summarize_marks_tiny_samples_unreliable():
    harmful, benign = _clusters(4, 16, 3, 5.0, 3)
    out = summarize_modality(harmful, benign, "video")
    assert out["reliable"] is False and out["linearly_encoded"] is False   # never claim from few samples
    assert "noisy" in out["reliability_note"]


def test_shape_validation():
    with pytest.raises(ValueError):
        summarize_modality([1, 2, 3], [[1, 2]], "image")          # 1D harmful
    with pytest.raises(ValueError):
        summarize_modality([[1, 2, 3]], [[1, 2]], "image")        # dim mismatch
    with pytest.raises(ValueError):
        summarize_modality([], [[1, 2]], "image")                 # empty side


def test_modalities_constant():
    assert set(MODALITIES) == {"image", "audio", "video"}
