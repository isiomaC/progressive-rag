"""Okapi BM25 sparse retrieval — implemented from scratch (educational).

score(q, d) = Σ idf(t) * f(t,d)*(k1+1) / (f(t,d) + k1*(1 - b + b*|d|/avgdl))

Term frequency is saturated by k1 (default 1.5); document length is
normalized by b (default 0.75); rare terms are boosted by IDF. Unlike
embeddings, BM25 matches exact tokens — identifiers, error codes, config
keys — which is why it complements vector search in hybrid retrieval.
"""

from __future__ import annotations

import math
from collections import Counter


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._tokens: list[list[str]] = []
        self._df: Counter = Counter()
        self._n = 0
        self._avgdl = 0.0

    def fit(self, texts: list[str]) -> "BM25Index":
        self._tokens = [text.lower().split() for text in texts]
        self._n = len(self._tokens)
        self._avgdl = sum(len(t) for t in self._tokens) / max(1, self._n)
        self._df = Counter()
        for toks in self._tokens:
            self._df.update(set(toks))
        return self

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        denom_factor = self.k1 * (1 - self.b + self.b * dl / self._avgdl)
        score = 0.0
        for t in set(query_tokens):
            if t not in self._df:
                continue
            f = tf.get(t, 0)
            score += self._idf(t) * (f * (self.k1 + 1)) / (f + denom_factor)
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        qt = [t for t in query.lower().split()]
        if not qt:
            return []
        scored = [(i, self.score(qt, toks)) for i, toks in enumerate(self._tokens)]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
