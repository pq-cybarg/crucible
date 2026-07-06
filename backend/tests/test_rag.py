"""Retrieval scoring: BM25 lexical (offline) + cosine semantic (injected embedder). Both label
their method honestly — a keyword hit is never dressed up as semantic relevance."""
import numpy as np

from crucible.rag import bm25_scores, cosine, rank, tokenize


DOCS = [
    "abliteration removes the refusal direction from the language model",
    "quantization compresses weights to run faster on local hardware",
    "the vision encoder handles image safety gates",
]


def test_tokenize_and_cosine():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]
    assert cosine([1, 0], [1, 0]) == 1.0
    assert abs(cosine([1, 0], [0, 1])) < 1e-9
    assert cosine([0, 0], [1, 1]) == 0.0            # zero vector -> 0, no div-by-zero


def test_bm25_ranks_keyword_match_first():
    scores = bm25_scores("refusal direction", DOCS)
    assert scores[0] > scores[1] and scores[0] > scores[2]   # doc 0 mentions both terms


def test_rank_lexical_is_labeled_and_ordered():
    r = rank("compress weights faster", DOCS, k=3)
    assert r["method"] == "lexical"
    assert r["results"][0]["index"] == 1             # the quantization doc
    assert all(res["score"] > 0 for res in r["results"])


def test_rank_semantic_uses_the_embedder():
    # a toy embedder: each doc -> a 3-dim one-hot by topic; query embeds to the same space
    space = {"refusal": [1, 0, 0], "compress": [0, 1, 0], "image": [0, 0, 1]}

    def embed(texts):
        out = []
        for t in texts:
            v = [0, 0, 0]
            for i, key in enumerate(("refusal", "compress", "image")):
                if key in t.lower():
                    v[i] = 1
            out.append(v)
        return out

    r = rank("image safety", DOCS, embedder=embed)
    assert r["method"] == "semantic"
    assert r["results"][0]["index"] == 2             # the vision-encoder doc, by cosine


def test_rank_empty_docs():
    assert rank("anything", [])["results"] == []
    assert rank("anything", [], embedder=lambda x: [[1]])["method"] == "semantic"


def test_rank_no_match_returns_empty():
    assert rank("xylophone quokka", DOCS)["results"] == []   # no shared terms -> nothing
