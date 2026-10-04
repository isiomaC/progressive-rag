"""Retrievers and rerankers — Phases 1, 3, and 6.1.

Every retriever returns a list of ScoredChunk, so variants are drop-in
swappable. Composition is the point: rerankers wrap base retrievers,
parent-document mapping wraps child retrieval, and the advanced combo
chains several stages together.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config, estimate_tokens
from .stores import CorpusIndex

RRF_K = 60


@dataclass
class ScoredChunk:
    id: str
    doc_id: str
    text: str
    score: float
    source: str = ""
    parent_id: str | None = None


def rrf_merge(lists: list[list], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score = Σ 1/(k + rank). Ranks, not raw scores,
    are comparable across heterogeneous indexes — that is the whole trick.
    Accepts lists of (id, score) pairs or plain id lists."""
    fused: dict[str, float] = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            cid = item[0] if isinstance(item, (tuple, list)) else item
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class Retriever:
    name = "base"

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        raise NotImplementedError


class VectorRetriever(Retriever):
    name = "vector"

    def __init__(self, index: CorpusIndex):
        self.index = index

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        return [self._wrap(cid, s, "vector") for cid, s in self.index.search_vector(question, k)]

    def _wrap(self, cid: str, s: float, source: str) -> ScoredChunk:
        c = self.index.get(cid)
        return ScoredChunk(c.id, c.doc_id, c.text, s, source, c.parent_id)


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, index: CorpusIndex):
        self.index = index

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        out = []
        for cid, s in self.index.search_bm25(question, k):
            c = self.index.get(cid)
            out.append(ScoredChunk(c.id, c.doc_id, c.text, s, "bm25", c.parent_id))
        return out


class HybridRetriever(Retriever):
    """Vector + BM25 fused with RRF. Covers both paraphrase and exact tokens."""

    name = "hybrid"

    def __init__(self, index: CorpusIndex):
        self.index = index

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        vec = self.index.search_vector(question, max(k * 4, 20))
        lex = self.index.search_bm25(question, max(k * 4, 20))
        fused = rrf_merge([vec, lex])[:k]
        out = []
        for cid, s in fused:
            c = self.index.get(cid)
            out.append(ScoredChunk(c.id, c.doc_id, c.text, s, "hybrid-rrf", c.parent_id))
        return out


class MultiQueryRetriever(Retriever):
    """LLM generates n reformulations; retrieve for each; RRF-merge the lists.

    Costs 1 LLM call + n retrieval passes. Shines when one formulation
    matches the corpus vocabulary and another does not.
    """

    name = "multi-query"

    def __init__(self, base: Retriever, llm, n_variants: int = 3):
        self.base, self.llm, self.n = base, llm, n_variants

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        variants = self.llm.generate_queries(question, self.n)
        lists = []
        seen: dict[str, ScoredChunk] = {}
        for v in variants:
            for sc in self.base.retrieve(v, k):
                if sc.id not in seen or sc.score > seen[sc.id].score:
                    seen[sc.id] = sc
        fused = rrf_merge([[c.id for c in self.base.retrieve(v, k)] for v in variants])[:k]
        return [seen[cid] for cid, _ in fused if cid in seen]


class HyDERetriever(Retriever):
    """Embed an LLM-written hypothetical answer instead of the raw question.

    The guessed answer is closer in vocabulary to real documents than the
    question is. Costs 1 LLM call; can hurt when the guess is confidently
    wrong.
    """

    name = "hyde"

    def __init__(self, index: CorpusIndex, llm):
        self.index, self.llm = index, llm

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        hyp = self.llm.hypothetical_answer(question)
        out = []
        for cid, s in self.index.search_vector(hyp, k):
            c = self.index.get(cid)
            out.append(ScoredChunk(c.id, c.doc_id, c.text, s, "hyde", c.parent_id))
        return out


class Reranker:
    name = "base"

    def rerank(self, question: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """Jointly encodes (query, chunk) pairs — far more accurate than the
    bi-encoder similarity that produced the candidates, too slow for the
    full corpus. Only reranks the candidate list."""

    name = "cross-encoder"

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def rerank(self, question: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        scores = self._model.predict([(question, c.text) for c in candidates])
        for sc, s in zip(candidates, scores):
            sc.score = float(s)
        return sorted(candidates, key=lambda c: c.score, reverse=True)


class LexicalReranker(Reranker):
    """Offline fallback: blend normalized BM25 and vector similarity.

    Weaker than a cross-encoder (cannot judge paraphrase quality) but
    recovers exact-token matches that the raw vector ranking missed."""

    name = "lexical"

    def __init__(self, index: CorpusIndex):
        self.index = index

    def rerank(self, question: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        bm25 = {c.id: s for c in self._wrap(self.index.search_bm25(question, len(candidates)))}
        vec = {c.id: s for c in self._wrap(self.index.search_vector(question, len(candidates)))}
        for sc in candidates:
            sc.score = _norm(bm25.get(sc.id, 0.0), 1) * 0.4 + _norm(vec.get(sc.id, 0.0), 2) * 0.6
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def _wrap(self, pairs):
        return [self._sc(cid, s) for cid, s in pairs]

    def _sc(self, cid, s):
        return ScoredChunk(cid, self.index.get(cid).doc_id, self.index.get(cid).text, s, "lexical")


def _norm(score: float, which: int) -> float:
    # BM25 scores are unbounded; vector scores are cosine in [-1, 1].
    # A soft normalization keeps both on a comparable 0..1-ish scale.
    import math
    if which == 1:
        return 1.0 - math.exp(-score)
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def build_reranker(cfg: Config, index: CorpusIndex) -> Reranker:
    try:
        r = CrossEncoderReranker(cfg.rerank_model_name)
        # warm up so failures surface now, not mid-experiment
        r.rerank("warm-up", [ScoredChunk("x", "d", "warm-up text", 0.0)])
        return r
    except Exception as exc:
        print(f"[reranker] cross-encoder unavailable ({type(exc).__name__}); "
              f"using lexical fallback reranker")
        return LexicalReranker(index)


class RerankRetriever(Retriever):
    """Two-stage retrieve-then-rerank: fetch candidates cheaply, rescore the
    short list expensively. The highest accuracy-per-millisecond upgrade."""

    name = "rerank"

    def __init__(self, base: Retriever, reranker: Reranker, candidates: int = 50):
        self.base, self.reranker, self.candidates = base, reranker, candidates

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        cands = self.base.retrieve(question, max(self.candidates, k))
        reranked = self.reranker.rerank(question, cands)
        for sc in reranked:
            sc.source = f"rerank[{self.reranker.name}]"
        return reranked[:k]


class ParentDocRetriever(Retriever):
    """Small-to-big: retrieve children for precision, return their parent
    chunks for context completeness. Deduplicates parents (several children
    share one parent)."""

    name = "parent-doc"

    def __init__(self, base: Retriever, index: CorpusIndex):
        self.base, self.index = base, index

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        children = self.base.retrieve(question, max(k * 3, 15))
        parents: dict[str, ScoredChunk] = {}
        for c in children:
            if not c.parent_id:
                continue
            p = self.index.chunk_by_id.get(c.parent_id)
            if p is None:
                continue
            best = parents.get(p.id)
            if best is None or c.score > best.score:
                parents[p.id] = ScoredChunk(p.id, p.doc_id, p.text, c.score, "parent-doc", None)
        return sorted(parents.values(), key=lambda x: x.score, reverse=True)[:k]


class AdvancedComboRetriever(Retriever):
    """Phase 6.1: query rewriting + multi-query + HyDE + rerank + parent-doc,
    composed. Each stage wraps the previous one."""

    name = "advanced-combo"

    def __init__(self, base: Retriever, reranker: Reranker, llm, index: CorpusIndex):
        self.reranker, self.llm, self.index = reranker, llm, index
        self.mq = MultiQueryRetriever(base, llm, n_variants=3)
        self.hyde = HyDERetriever(index, llm)
        self.rr = RerankRetriever(self.mq, reranker)
        self.parent = ParentDocRetriever(self.mq, index)

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        rewritten = self.llm.rewrite(question)
        lists = [
            self.mq.retrieve(rewritten, k),
            self.hyde.retrieve(rewritten, k),
        ]
        fused = rrf_merge([[c.id for c in lst] for lst in lists])
        seen = {c.id: c for lst in lists for c in lst}
        cands = [seen[cid] for cid, _ in fused[:50] if cid in seen]
        reranked = self.reranker.rerank(rewritten, cands)[:k]
        # parent expansion on the winning chunks (if they are children)
        parents: dict[str, ScoredChunk] = {}
        for c in reranked:
            if c.parent_id:
                p = self.index.chunk_by_id.get(c.parent_id)
                if p:
                    parents[p.id] = ScoredChunk(p.id, p.doc_id, p.text, c.score, "advanced-combo", None)
        if parents:
            return sorted(parents.values(), key=lambda x: x.score, reverse=True)[:k]
        return reranked
