# Architecture

## Design principles

1. **Everything is a swappable component.** Six small interfaces — `Chunker`,
   `Embedder`, `Retriever`, `Reranker`, `LLM`, `Generator` — with a stable
   data type (`Chunk`, `ScoredChunk`, `CorpusIndex`) flowing between them.
   An experiment is a named wiring of components; changing chunking or
   retrieval never touches the rest of the pipeline.
2. **Determinism.** Every run reads the same corpus + golden set and writes
   `experiments/results.json`. Embeddings are cached per (model, text), so
   reruns are fast and reproducible.
3. **Two runtime modes, one code path.** `offline` (no keys: local
   embeddings + extractive generator + lexical heuristics) and online
   (`openai` / `deepseek` providers). The report labels which mode produced
   the numbers.

## Data flow

```
data/corpus/*.md ──load──> Document[] ──chunk──> Chunk[]
                                                     │
                              CorpusIndex ──────────┼── VectorStore (cosine)
                              (one per strategy)    └── BM25Index (Okapi)
                                                     │
question ──Retriever──> ScoredChunk[] ──Generator──> GenerationResult
                              │
                              ├── rerankers wrap retrievers
                              ├── ParentDocRetriever maps children → parents
                              └── agents wrap the whole loop
```

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | paths, chunk sizes, `.env` loading, pricing table, token estimation |
| `corpus.py` | markdown → `Document` with sections; sentence splitting |
| `chunking.py` | `FixedChunker`, `RecursiveChunker`, `SemanticChunker`, `ParentChildChunker` |
| `embeddings.py` | `SentenceTransformerEmbedder` (real) + `HashingTFIDFEmbedder` (fallback) + disk cache |
| `bm25.py` | Okapi BM25 from scratch (k1=1.5, b=0.75) |
| `stores.py` | `VectorStore` (exact cosine, numpy) + `CorpusIndex` (registry + both indexes) |
| `retrieval.py` | vector / bm25 / hybrid(RRF) / multi-query / HyDE / rerank / parent-doc / advanced combo + rerankers |
| `llm.py` | `OpenAICompatibleLLM` (OpenAI + DeepSeek), `OfflineLLM`, usage tracking |
| `generation.py` | RAG prompt template, `Generator` (LLM), `ExtractiveGenerator` (offline) |
| `eval_metrics.py` | recall@k, hit rate, MRR, NDCG, context P/R, faithfulness/relevancy/correctness proxies |
| `golden.py` | YAML golden set + per-strategy relevant-chunk resolution |
| `graph.py` | entity extraction, co-occurrence graph, `GraphRetriever` |
| `agents.py` | `AgenticRAG`, `SelfRAG` decision loops |
| `experiments.py` | variant definitions + evaluation harness + Phase 5 challenges |
| `cli.py` | `ragprog run / report / smoke` |

## The CorpusIndex invariant

For every chunking strategy we build ONE `CorpusIndex`:

- chunks → embeddings (cached) → `VectorStore`
- chunks → `BM25Index`
- `chunk_by_id` registry (extended with parents for parent-child strategies)

Retrieval variants differ only in how they query the index — which is what
makes the comparison fair: same documents, same embedder, one variable
changed at a time.

## Evaluation contract

`_evaluate_variant` scores every variant over the same 51 golden questions:

- retrieval: chunk-level (vs answer-overlap-resolved relevant chunks) and
  document-level (vs golden `docs`),
- generation: faithfulness / relevancy / correctness (lexical proxies always;
  LLM judges when online),
- latency: retrieval and generation timed separately per question,
- cost: token deltas attributed per question from the LLM usage ledger.

Per-type aggregation (single-hop / multi-hop / tricky / comparative) is what
makes the multi-hop and "I don't know" findings visible.

## Why the golden set is resolved per chunking strategy

Chunk boundaries differ between strategies, so "the relevant chunk" cannot
be a single stable id. `resolve_relevant_chunks` re-derives, per strategy,
the chunk with the highest overlap with the golden answer in each required
document. Document-level metrics are strategy-independent and are the
primary comparison axis.
