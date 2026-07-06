from __future__ import annotations
# Retrieval scoring for knowledge / memory. Two HONEST modes:
#   - SEMANTIC: cosine similarity over real embeddings from a configured embedding model. The
#     embedder is injected, so the math is pure + unit-tested; the endpoint supplies a real embedder
#     or says plainly that none is available (it never fabricates vectors).
#   - LEXICAL: BM25 term scoring — no model needed, works offline — but it's KEYWORD matching, not
#     meaning. Every result is LABELED with its method so a keyword hit is never mistaken for
#     semantic relevance (no-fake-metrics).
import math
import re
from collections import Counter
from typing import Callable, Optional

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 relevance of each doc to the query — classic lexical ranking (term frequency × inverse
    document frequency, length-normalized). Keyword matching, not semantics."""
    doc_tokens = [tokenize(d) for d in docs]
    n = len(docs) or 1
    avgdl = (sum(len(t) for t in doc_tokens) / n) or 1.0
    df: Counter = Counter()
    for toks in doc_tokens:
        for term in set(toks):
            df[term] += 1
    q_terms = tokenize(query)
    scores = []
    for toks in doc_tokens:
        tf = Counter(toks)
        dl = len(toks) or 1
        s = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * (tf[term] * (k1 + 1)) / (tf[term] + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def rank(query: str, docs: list[str], k: int = 5,
         embedder: Optional[Callable[[list[str]], list]] = None,
         query_embedding=None) -> dict:
    """Rank docs by relevance to the query. With an embedder (texts -> list[vector]) it's SEMANTIC
    (cosine); without one it's LEXICAL (BM25). Returns {method, results:[{index, score}]} for the
    top-k with a positive score — the honest method is always reported."""
    if not docs:
        return {"method": "semantic" if embedder else "lexical", "results": []}
    if embedder is not None:
        vecs = embedder(docs)
        qv = query_embedding if query_embedding is not None else embedder([query])[0]
        scores = [cosine(qv, v) for v in vecs]
        method = "semantic"
    else:
        scores = bm25_scores(query, docs)
        method = "lexical"
    order = sorted(range(len(docs)), key=lambda i: -scores[i])[:k]
    results = [{"index": i, "score": round(float(scores[i]), 4)} for i in order if scores[i] > 0]
    return {"method": method, "results": results}
