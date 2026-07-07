from __future__ import annotations
# Pluggable DISTANCE / SIMILARITY measures for organizing memory & context. One registry, four honest
# families — so recall, search, clustering and re-crystallization can all pick how "close" two pieces
# of knowledge are, and every result carries a label that says WHICH kind of closeness it measured
# (no-fake-metrics: a keyword hit, a bag-of-words statistic, a learned embedding, and a model's opinion
# are NOT the same thing and are never presented as if they were):
#
#   statistical  — jaccard / dice / overlap / tf-idf cosine / edit-distance. Pure bag-of-words or
#                  character math. Offline, deterministic, cheap. NOT meaning — token/char overlap.
#   lexical      — BM25. Classic tf·idf ranking. Also keyword matching, not meaning.
#   semantic     — cosine over REAL embeddings from a configured embedding model (injected). Meaning,
#                  but only as good as the embedder; unavailable when no embedder is configured.
#   llm-judged   — a (preferably small, cheap) "processing" model rates relatedness. This is the
#                  brain-plasticity / preprocessing role: a background reorganizer's OPINION. Labeled
#                  as a subjective model judgment, never as a rigorous metric.
#
# Every scorer returns a per-doc similarity in [0, 1] (higher = closer) so the families are comparable
# and can feed the same relevance sort. `score()` reports the honest method label alongside.
import re
from collections import Counter
from typing import Callable, Optional

from crucible.rag import bm25_scores, cosine, tokenize

# name -> honest method label surfaced to the user. The label is the promise: it names the KIND of
# closeness, so nothing statistical is ever dressed up as semantic and no model opinion as a metric.
LABELS: dict[str, str] = {
    "bm25": "lexical-bm25",
    "jaccard": "statistical-jaccard",
    "dice": "statistical-dice",
    "overlap": "statistical-overlap",
    "tfidf": "statistical-tfidf-cosine",
    "edit": "statistical-edit-distance",
    "embedding": "semantic-embedding",
    "llm": "llm-judged",
}
# families that need no model/embedder — always available, fully offline & deterministic.
OFFLINE = ("bm25", "jaccard", "dice", "overlap", "tfidf", "edit")
METRICS = tuple(LABELS)


def _minmax(values: list[float]) -> list[float]:
    """Scale to [0,1] so heterogeneous raw scales (BM25 sums, edit counts, cosines) stay comparable.
    A flat set maps to all-zeros (nothing stands out) rather than dividing by zero."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:
        return [0.0 for _ in values]
    return [(v - lo) / span for v in values]


# --- set-overlap statistics (bag of tokens) ---------------------------------------------------------
def _sets(query: str, docs: list[str]) -> tuple[set[str], list[set[str]]]:
    return set(tokenize(query)), [set(tokenize(d)) for d in docs]


def jaccard_scores(query: str, docs: list[str]) -> list[float]:
    q, ds = _sets(query, docs)
    return [(len(q & d) / len(q | d)) if (q or d) else 0.0 for d in ds]


def dice_scores(query: str, docs: list[str]) -> list[float]:
    q, ds = _sets(query, docs)
    return [(2 * len(q & d) / (len(q) + len(d))) if (q or d) else 0.0 for d in ds]


def overlap_scores(query: str, docs: list[str]) -> list[float]:
    """Szymkiewicz–Simpson overlap coefficient: |A∩B| / min(|A|,|B|) — robust to size mismatch."""
    q, ds = _sets(query, docs)
    return [(len(q & d) / min(len(q), len(d))) if (q and d) else 0.0 for d in ds]


# --- tf-idf cosine (statistical vector-space, no learned model) -------------------------------------
def tfidf_cosine_scores(query: str, docs: list[str]) -> list[float]:
    """Cosine over tf·idf bag-of-words vectors built from THIS doc set. Statistical, not learned —
    it captures term co-occurrence, not meaning, and is labeled accordingly."""
    import math

    doc_toks = [tokenize(d) for d in docs]
    n = len(docs) or 1
    df: Counter = Counter()
    for toks in doc_toks:
        for term in set(toks):
            df[term] += 1
    idf = {t: math.log((1 + n) / (1 + df[t])) + 1.0 for t in df}

    def vec(toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        return {t: tf[t] * idf.get(t, math.log(1 + n) + 1.0) for t in tf}

    qv = vec(tokenize(query))

    def cos(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b.get(t, 0.0) for t in a)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    return [cos(qv, vec(toks)) for toks in doc_toks]


# --- normalized edit distance (character-level) -----------------------------------------------------
def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_scores(query: str, docs: list[str]) -> list[float]:
    """1 − normalized Levenshtein distance. Character similarity — good for near-duplicates / typos,
    blind to meaning."""
    q = (query or "").lower()
    out = []
    for d in docs:
        t = (d or "").lower()
        m = max(len(q), len(t))
        out.append(1.0 - _levenshtein(q, t) / m if m else 1.0)
    return out


# --- embeddings (semantic) --------------------------------------------------------------------------
def embedding_scores(query: str, docs: list[str], embedder: Callable[[list[str]], list]) -> list[float]:
    vecs = embedder(docs)
    qv = embedder([query])[0]
    return [cosine(qv, v) for v in vecs]


# --- llm-judged (a small processing model's opinion) ------------------------------------------------
_INT = re.compile(r"-?\d+")


def llm_scores(query: str, docs: list[str], solver: Callable[[str], str]) -> list[float]:
    """Ask a (small, cheap) processing model to rate 0–100 how related each doc is to the query, then
    scale to [0,1]. This is the model's OPINION — a plasticity/reorganization signal, not a rigorous
    metric — and is labeled 'llm-judged'. A garbled reply scores 0 (never fabricated)."""
    out = []
    for d in docs:
        prompt = (
            "Rate how RELATED these two texts are on a 0-100 scale (100 = same topic/meaning, "
            "0 = unrelated). Reply with ONLY the integer.\n\n"
            f"A: {query}\n\nB: {d}\n\nScore:"
        )
        try:
            reply = solver(prompt) or ""
            m = _INT.search(reply)
            val = max(0.0, min(100.0, float(m.group()))) / 100.0 if m else 0.0
        except Exception:
            val = 0.0
        out.append(val)
    return out


def available(metric: str, embedder=None, solver=None) -> bool:
    """Can this metric run with the resources at hand? Offline ones always; embedding needs an
    embedder; llm needs a solver (the processing model)."""
    if metric in OFFLINE:
        return True
    if metric == "embedding":
        return embedder is not None
    if metric == "llm":
        return solver is not None
    return False


def score(query: str, docs: list[str], metric: str = "bm25",
          embedder: Optional[Callable[[list[str]], list]] = None,
          solver: Optional[Callable[[str], str]] = None,
          normalize: bool = True) -> dict:
    """Score every doc against the query with the named metric. Returns {method, scores} where method
    is the HONEST label and scores are per-doc similarities (normalized to [0,1] when `normalize`).
    Raises ValueError for an unknown metric or one whose resource (embedder/solver) is missing."""
    if metric not in LABELS:
        raise ValueError(f"unknown metric '{metric}' (have: {', '.join(METRICS)})")
    if not available(metric, embedder, solver):
        need = "an embedding backend" if metric == "embedding" else "a processing model"
        raise ValueError(f"metric '{metric}' needs {need}, which is not configured")
    if not docs:
        return {"method": LABELS[metric], "scores": []}
    if metric == "bm25":
        raw = bm25_scores(query, docs)
    elif metric == "jaccard":
        raw = jaccard_scores(query, docs)
    elif metric == "dice":
        raw = dice_scores(query, docs)
    elif metric == "overlap":
        raw = overlap_scores(query, docs)
    elif metric == "tfidf":
        raw = tfidf_cosine_scores(query, docs)
    elif metric == "edit":
        raw = edit_scores(query, docs)
    elif metric == "embedding":
        raw = embedding_scores(query, docs, embedder)  # type: ignore[arg-type]
    else:  # llm
        raw = llm_scores(query, docs, solver)          # type: ignore[arg-type]
    scores = _minmax([float(x) for x in raw]) if normalize else [float(x) for x in raw]
    return {"method": LABELS[metric], "scores": scores}


def rank(query: str, docs: list[str], k: int = 5, metric: str = "bm25",
         embedder=None, solver=None) -> dict:
    """Top-k docs by the chosen metric. Returns {method, results:[{index, score}]} for positive
    scores — the honest method label always travels with the ranking."""
    res = score(query, docs, metric=metric, embedder=embedder, solver=solver)
    order = sorted(range(len(docs)), key=lambda i: -res["scores"][i])[:k]
    results = [{"index": i, "score": round(res["scores"][i], 4)} for i in order if res["scores"][i] > 0]
    return {"method": res["method"], "results": results}
