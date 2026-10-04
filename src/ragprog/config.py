"""Central configuration: paths, model choices, env loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (token-ish) cost per 1M tokens: (input, output). Documented estimates.
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "deepseek-chat": (0.27, 1.10),
}

DEFAULT_CHUNK = dict(
    # Sizes are scaled to this small (~10k word) corpus. Production corpora
    # typically use 500/50; here 100/20 gives ~150+ chunks to compare over.
    fixed_size=100,
    fixed_overlap=20,
    recursive_size=100,
    recursive_overlap=20,
    semantic_percentile=90,
    child_size=50,
    parent_size=250,
    retrieval_k=5,
    rerank_candidates=50,
)

# A small glossary that seeds the knowledge graph with entities that
# heuristic extraction (capitalized n-grams) would miss or mangle.
GRAPH_GLOSSARY = [
    "RAG", "BM25", "HyDE", "RRF", "HNSW", "IVF", "FAISS", "NDCG", "MRR",
    "RAGAS", "CRAG", "Self-RAG", "GraphRAG", "MiniLM", "cross-encoder",
    "bi-encoder", "reciprocal rank fusion", "small-to-big", "multi-hop",
    "chunking", "reranking", "multi-query", "query rewriting", "Leiden",
    "knowledge graph", "vector store", "embedding", "golden set",
]


@dataclass
class Config:
    data_dir: Path = ROOT / "data"
    corpus_dir: Path = ROOT / "data" / "corpus"
    golden_yaml: Path = ROOT / "data" / "golden" / "questions.yaml"
    cache_dir: Path = ROOT / "data" / "cache"
    results_file: Path = ROOT / "experiments" / "results.json"
    site_dir: Path = ROOT / "site"

    llm_provider: str = "offline"          # offline | openai | deepseek
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    embed_model_name: str = "all-MiniLM-L6-v2"
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk: dict = field(default_factory=lambda: dict(DEFAULT_CHUNK))

    def ensure_dirs(self) -> None:
        for d in (self.cache_dir, self.results_file.parent, self.site_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_env_file(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env (no dependency needed)."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def from_env() -> Config:
    load_env_file()
    return Config(
        llm_provider=os.environ.get("RAGPROG_LLM", "offline").lower(),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        deepseek_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        embed_model_name=os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2"),
        rerank_model_name=os.environ.get("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )


def words(text: str) -> int:
    """Cheap 'token' proxy: whitespace-separated words."""
    return len(text.split())


def estimate_tokens(text: str) -> int:
    """Words → approximate tokens (English text averages ~1.33 tokens/word)."""
    return int(words(text) * 1.33)
