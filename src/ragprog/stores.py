"""Stores: in-memory vector index + BM25 index over a chunk list.

CorpusIndex is the central data structure every retriever reads from.
It is rebuilt per chunking strategy: same documents, different chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import BM25Index
from .chunking import Chunk
from .embeddings import Embedder


class VectorStore:
    """Exact cosine search over normalized vectors (numpy matrix)."""

    def __init__(self, embeddings: np.ndarray, ids: list[str]):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = embeddings / norms
        self.ids = ids

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        q = query_vec / (np.linalg.norm(query_vec) or 1.0)
        sims = self.matrix @ q
        idx = np.argsort(-sims)[:top_k]
        return [(self.ids[i], float(sims[i])) for i in idx]

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class CorpusIndex:
    """Chunk registry + both indexes + embedder, for one chunking strategy."""

    chunks: list[Chunk]
    chunk_by_id: dict[str, Chunk]
    embedder: Embedder
    vector: VectorStore
    bm25: BM25Index

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "CorpusIndex":
        vecs = embedder.embed([c.text for c in chunks])
        return cls(
            chunks=chunks,
            chunk_by_id={c.id: c for c in chunks},
            embedder=embedder,
            vector=VectorStore(vecs, [c.id for c in chunks]),
            bm25=BM25Index().fit([c.text for c in chunks]),
        )

    def search_vector(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.vector.search(self.embedder.embed_query(query), k)

    def search_bm25(self, query: str, k: int) -> list[tuple[str, float]]:
        return [(self.chunks[i].id, s) for i, s in self.bm25.search(query, k)]

    def get(self, chunk_id: str) -> Chunk:
        return self.chunk_by_id[chunk_id]

    def __len__(self) -> int:
        return len(self.chunks)
