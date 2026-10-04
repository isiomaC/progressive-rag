"""Embeddings: real sentence-transformers with an offline fallback.

Embedder contract:
    embed(texts) -> np.ndarray   (normalized, unit-length rows)
    embed_query(text) -> np.ndarray
    dim -> int
    name -> str

Caching: embeddings are memoized per (model, text) on disk so reruns of
the experiment matrix are fast and deterministic.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

from .config import Config


class Embedder:
    name = "abstract"
    dim = 0
    semantic = False

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class SentenceTransformerEmbedder(Embedder):
    """Local bi-encoder (default: all-MiniLM-L6-v2, 384 dims)."""

    name = "sentence-transformers"
    semantic = True

    def __init__(self, model_name: str, cache_dir: Path):
        self.model_name = model_name
        self._cache = _EmbeddingCache(cache_dir, f"emb_{_slug(model_name)}")
        self._model = None

    @property
    def dim(self) -> int:
        if self._model is None:
            self._load()
        return int(self._model.get_sentence_embedding_dimension())

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        cached, missing, order = self._cache.lookup(texts)
        if missing:
            vecs = self._load().encode(
                missing, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
            self._cache.store(missing, vecs)
            cached = {t: self._cache.data[t] for t in order}   # refresh after store
        return np.vstack([cached[t] for t in order])


class HashingTFIDFEmbedder(Embedder):
    """Offline fallback: lexical n-gram hashing weighted by IDF.

    NOT a semantic embedder — it is a lower bound used when
    sentence-transformers is unavailable. Results in this mode are labeled
    as lexical in the report.
    """

    name = "hashing-tfidf"
    semantic = False

    def __init__(self, dim: int = 768, n: int = 2, vocab_min_df: int = 2):
        self._dim, self._n, self._min_df = dim, n, vocab_min_df
        self._idf: dict[int, float] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, texts: list[str]) -> "HashingTFIDFEmbedder":
        df: dict[int, int] = {}
        for text in texts:
            for g in set(_ngrams(text.lower(), self._n)):
                df[g] = df.get(g, 0) + 1
        n_docs = max(1, len(texts))
        self._idf = {
            g: np.log((n_docs + 1) / (d + 1)) + 1.0
            for g, d in df.items() if d >= self._min_df
        }
        return self

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        grams = _ngrams(text.lower(), self._n)
        if not grams:
            return vec
        for g in grams:
            w = self._idf.get(g, 0.0)
            if w:
                vec[g % self._dim] += w
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vectorize(t) for t in texts]).astype(np.float32)


def build_embedder(cfg: Config, corpus_texts: list[str] | None = None) -> Embedder:
    """Try the real model; fall back to hashing-TFIDF if unavailable."""
    try:
        embedder = SentenceTransformerEmbedder(cfg.embed_model_name, cfg.cache_dir)
        embedder.embed(["warm-up check"])   # forces a load now
        return embedder
    except Exception as exc:  # pragma: no cover — environment dependent
        print(f"[embedder] sentence-transformers unavailable ({type(exc).__name__}); "
              f"falling back to hashing-TFIDF (lexical mode)")
        return HashingTFIDFEmbedder().fit(corpus_texts or [])


def _ngrams(text: str, n: int) -> list[int]:
    words = text.split()
    grams = []
    for size in range(1, n + 1):
        grams.extend(
            hash(" ".join(words[i:i + size]))
            for i in range(len(words) - size + 1)
        )
    return grams


def _slug(name: str) -> str:
    return hashlib.md5(name.encode()).hexdigest()[:12]


class _EmbeddingCache:
    """On-disk memoization of (text → vector)."""

    def __init__(self, cache_dir: Path, key: str):
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = cache_dir / f"{key}.pkl"
        self.data: dict[str, np.ndarray] = {}
        if self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    self.data = pickle.load(f)
            except Exception:
                self.data = {}

    def lookup(self, texts: list[str]):
        cached = {t: self.data[t] for t in texts if t in self.data}
        missing = list(dict.fromkeys(t for t in texts if t not in self.data))
        return cached, missing, texts

    def store(self, texts: list[str], vecs: np.ndarray) -> None:
        for t, v in zip(texts, vecs):
            self.data[t] = v
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)
