"""Pluggable distance/similarity families — statistical, lexical, embedding, llm-judged. Every scorer
returns a normalized similarity and an HONEST method label (no-fake-metrics)."""
import numpy as np
import pytest

from crucible import metrics

DOCS = [
    "abliteration removes the refusal direction from the weights",
    "quantization compresses weights for faster inference",
    "the refusal direction is an orthogonal projection",
]


def test_offline_metrics_always_available_and_labeled():
    for m in metrics.OFFLINE:
        assert metrics.available(m)
        res = metrics.score("refusal direction", DOCS, metric=m)
        assert res["method"] == metrics.LABELS[m]
        assert len(res["scores"]) == len(DOCS)
        assert all(0.0 <= s <= 1.0 for s in res["scores"])   # normalized


def test_statistical_metrics_rank_topically_related_first():
    for m in ("bm25", "jaccard", "dice", "overlap", "tfidf"):
        r = metrics.rank("refusal direction", DOCS, metric=m)
        assert r["results"], f"{m} found nothing"
        top = r["results"][0]["index"]
        assert top in (0, 2)      # the two docs about the refusal direction, not the quantization one


def test_edit_distance_is_character_similarity():
    r = metrics.score("quantization compresses weights for faster inference", DOCS, metric="edit")
    assert r["method"] == "statistical-edit-distance"
    assert r["scores"][1] == max(r["scores"])   # the near-identical doc wins on characters


def test_embedding_metric_uses_injected_embedder():
    # a toy deterministic embedder: bag-of-words over a tiny vocab → real cosine, no fabrication
    vocab = ["refusal", "direction", "quantization", "weights"]

    def embedder(texts):
        return [[float(t.lower().count(w)) for w in vocab] for t in texts]

    assert metrics.available("embedding", embedder=embedder)
    r = metrics.rank("refusal direction", DOCS, metric="embedding", embedder=embedder)
    assert r["method"] == "semantic-embedding"
    assert r["results"][0]["index"] in (0, 2)


def test_embedding_unavailable_without_embedder():
    assert not metrics.available("embedding")
    with pytest.raises(ValueError):
        metrics.score("x", DOCS, metric="embedding")


def test_llm_metric_is_the_processing_models_opinion():
    # a stub processing model that "rates" by keyword presence — labeled llm-judged, never a metric
    def solver(prompt: str) -> str:
        return "90" if "refusal" in prompt.lower() else "10"

    assert metrics.available("llm", solver=solver)
    r = metrics.score("refusal", DOCS, metric="llm", solver=solver)
    assert r["method"] == "llm-judged"
    assert r["scores"][0] == max(r["scores"])   # the refusal docs score high
    # a garbled reply scores 0, never fabricated
    assert metrics.score("x", ["y"], metric="llm", solver=lambda p: "not a number")["scores"] == [0.0]


def test_llm_unavailable_without_solver():
    assert not metrics.available("llm")
    with pytest.raises(ValueError):
        metrics.score("x", DOCS, metric="llm")


def test_unknown_metric_raises_and_empty_docs_is_safe():
    with pytest.raises(ValueError):
        metrics.score("x", DOCS, metric="telepathy")
    assert metrics.score("x", [], metric="bm25") == {"method": "lexical-bm25", "scores": []}
