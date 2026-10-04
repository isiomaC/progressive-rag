#!/usr/bin/env python
"""Quick smoke test of the whole pipeline (no experiments)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragprog.config import from_env
from ragprog.corpus import load_corpus, split_sentences
from ragprog.chunking import FixedChunker, RecursiveChunker, SemanticChunker, ParentChildChunker
from ragprog.embeddings import build_embedder
from ragprog.golden import load_golden, resolve_relevant_chunks
from ragprog.stores import CorpusIndex

cfg = from_env()
cfg.ensure_dirs()
docs = load_corpus(cfg.corpus_dir)
golden = load_golden(cfg.golden_yaml)
print(f"corpus: {len(docs)} docs, {sum(d.word_count for d in docs)} words")
print(f"golden: {len(golden)} questions")

emb = build_embedder(cfg, [d.text for d in docs])
print(f"embedder: {emb.name} ({emb.dim}d, semantic={emb.semantic})")

for chunker in [FixedChunker(100, 20), RecursiveChunker(100, 20),
                SemanticChunker(emb, 90), ParentChildChunker(50, 250)]:
    chunks = [c for d in docs for c in chunker.chunk(d)]
    print(f"  {chunker.name:14s} -> {len(chunks):4d} chunks")

idx = CorpusIndex.build([c for d in docs for c in RecursiveChunker(100, 20).chunk(d)], emb)
rel = resolve_relevant_chunks(golden, idx.chunks)
print("relevant chunk resolution: ok (", sum(len(v) for v in rel.values()), "chunks mapped )")

q = golden[0]
hits = idx.search_vector(q.question, 5)
print(f"sample q [{q.id}]: {q.question}")
print("  top docs:", [idx.get(c).doc_id for c, _ in hits])
print("  relevant docs:", q.docs)
