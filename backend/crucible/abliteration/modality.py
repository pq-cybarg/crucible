from __future__ import annotations
# Modality refusal/safety directions. Text refusal is a direction in the language model's residual
# stream; an IMAGE or AUDIO safety gate is a direction in the ENCODER'S embedding space. The math is
# the same contrastive diff-of-means (harmful vs benign, per modality) — but it MUST be measured on
# embeddings from that modality's encoder (run harmful vs benign images/audio through the model's
# own tower, or CLIP / whisper). This computes the direction + how linearly separable the safety
# concept is in that space, so the encoder/connector can be orthogonalized against it (part-scoped).
#
# HONEST BY CONSTRUCTION: it does NOT fabricate embeddings. The caller supplies real modality
# vectors (or a multimodal adapter provides them). With none available, the endpoint says so plainly
# rather than inventing numbers — matching the composition report's executable_now vs needs_probing.
import numpy as np
from numpy.typing import ArrayLike

from crucible.abliteration.concept import concept_vector, separability

MODALITIES = ("image", "audio", "video")


def modality_direction(harmful: ArrayLike, benign: ArrayLike) -> np.ndarray:
    """Unit safety/refusal direction in a modality's embedding space: normalized
    mean(harmful) - mean(benign). Same contrastive math as text refusal, on encoder embeddings."""
    return concept_vector(harmful, benign, normalize=True)


def held_out_separability(harmful: ArrayLike, benign: ArrayLike):
    """2-fold cross-validated separability: fit the direction on a TRAIN half, measure Cohen's-d
    separation on the HELD-OUT half (both folds, averaged). This is the honest number — in-sample
    separability along the diff-of-means direction is optimistically biased and makes even two
    random clusters look separated when dim > n. Returns None if there aren't enough samples to
    hold any out (< 2 per side)."""
    h = np.asarray(harmful, dtype=np.float64)
    b = np.asarray(benign, dtype=np.float64)
    he, ho, be, bo = h[::2], h[1::2], b[::2], b[1::2]
    if min(len(he), len(ho), len(be), len(bo)) == 0:
        return None

    def fold(h_tr, h_te, b_tr, b_te) -> float:
        d = concept_vector(h_tr, b_tr, normalize=True)
        return float(separability(h_te, b_te, d))

    return round((fold(he, ho, be, bo) + fold(ho, he, bo, be)) / 2.0, 4)


def summarize_modality(harmful: ArrayLike, benign: ArrayLike, modality: str = "image") -> dict:
    """Compute the modality direction and report how linearly encoded the safety concept is
    (Cohen's-d separability) — the honest read on whether an encoder-space edit will actually work.
    Validates shapes; raises ValueError on mismatched/empty embeddings."""
    h = np.asarray(harmful, dtype=np.float64)
    b = np.asarray(benign, dtype=np.float64)
    if h.ndim != 2 or b.ndim != 2:
        raise ValueError("harmful and benign must each be a 2D array of embeddings (n x dim)")
    if h.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError("need at least one embedding on each side")
    if h.shape[1] != b.shape[1]:
        raise ValueError(f"embedding dim mismatch: harmful {h.shape[1]} vs benign {b.shape[1]}")
    direction = modality_direction(h, b)
    in_sample = float(separability(h, b, direction))
    held_out = held_out_separability(h, b)
    # The HELD-OUT number is the honest one (unbiased — it's ~0 for random clusters, unlike
    # in-sample which is always positive when dim > n). Fall back to in-sample only when there
    # aren't enough samples to split, and say so. A held-out estimate is noisy at small n, so we
    # flag `reliable` on sample size and never claim "encoded" from an unreliable read.
    honest = held_out if held_out is not None else in_sample
    min_side = int(min(h.shape[0], b.shape[0]))
    reliable = held_out is not None and min_side >= 10
    return {
        "modality": modality,
        "n_harmful": int(h.shape[0]),
        "n_benign": int(b.shape[0]),
        "dim": int(h.shape[1]),
        "separability": round(float(honest), 4),
        "separability_kind": "held-out (2-fold cross-validated)" if held_out is not None
                             else "in-sample (too few to hold out — optimistic)",
        "in_sample_separability": round(in_sample, 4),
        "reliable": reliable,
        "reliability_note": ("held-out estimate over adequate samples"
                             if reliable else f"noisy — only {min_side} per side; use >= ~10 for a confident read"),
        "linearly_encoded": bool(reliable and abs(honest) >= 0.5),
        "direction_norm": 1.0,
        "direction": direction.tolist(),
    }
