# Implementation Plan

This is the plan that was executed for the **Progressive RAG Documentation
Assistant** learning project. Every phase of the original brief is mapped to
code, experiments, and documentation.

## Architecture decision (up front)

- **Language/runtime:** Python 3.13, minimal dependencies (`numpy` core;
  `sentence-transformers` optional for real embeddings + cross-encoder rerank).
- **Everything is a pluggable component** behind small interfaces:
  `Chunker`, `Embedder`, `Retriever`, `LLM`, `Generator`. Experiments are
  data-driven (a registry of chunk→embedding→index), so swapping a component
  never requires rewriting the pipeline.
- **Two runtime modes** (same code path):
  - *Offline mode* (default, no API key): local embeddings + a deterministic
    extractive generator + heuristic query transforms. Lets the whole project
    run anywhere for free.
  - *Online mode* (`OPENAI_API_KEY` in `.env`): real LLM answers, HyDE,
    multi-query, critiques, and LLM-judge metrics. Set `LLM_MODEL` to target
    any OpenAI-compatible endpoint (Ollama, Groq, …).
- **Deterministic experiments:** every run writes `experiments/*.json` with
  retrieval + generation metrics, latency per component, and token/cost
  estimates, so numbers are always reproducible and comparable.

## Phase mapping

| Phase | What we build | Where |
|---|---|---|
| 1 — Naive RAG baseline | fixed chunking + vector top-k + simple context/question prompt + hit-rate eval | `chunking.FixedChunker`, `retrieval.VectorRetriever`, `generation.RAG_PROMPT`, variant `baseline` |
| 2 — Chunking experiments | fixed / recursive / semantic / parent-child (small-to-big) — 4-way comparison on retrieval + answer quality | `chunking.*`, variant group `chunking` |
| 3 — Advanced retrieval | hybrid (RRF), multi-query, HyDE, reranker (cross-encoder, lexical fallback) + latency/cost measurement | `retrieval.*`, variant group `retrieval` |
| 4 — Evaluation framework | Recall@k, Precision@k, HitRate, MRR, NDCG, Context P/R; faithfulness, answer relevancy, correctness (RAGAS-style, implemented locally + LLM-judge when online); golden dataset of 50 questions with ground-truth answers + relevant chunks; bad-retrieval detector | `eval_metrics.py`, `data/golden/` |
| 5 — Practical challenges lab | forced bad retrieval, per-component latency, cost tracking, KB add/delete/update, multi-hop question analysis | `experiments.challenges`, variant group `challenges` |
| 6.1 — Advanced RAG | query rewrite + HyDE + multi-query + rerank + parent-child, composed | `retrieval.AdvancedComboRetriever` |
| 6.2 — GraphRAG | entities + co-occurrence graph + graph-guided retrieval (LlamaIndex-KG style, explained against Microsoft GraphRAG) | `graph.py` |
| 6.3 — Agentic RAG | route (need retrieval?), rewrite, multi-step retrieval loop, stop conditions | `agents.AgenticRAG` |
| 6.4 — Corrective / Self-RAG | generate → critique context support → re-retrieve or answer "I don't know" | `agents.SelfRAG` |

## Deliverable mapping

1. **Modular codebase** → `src/ragprog/` (see `docs/ARCHITECTURE.md`).
2. **Comparison table / dashboard** → `site/index.html` (self-contained visual
   report with charts built from `experiments/results.json`).
3. **Evaluation report with numbers** → `docs/FINDINGS.md` + the HTML dashboard.
4. **Short write-up** → `docs/FINDINGS.md` answers all 5 questions
   (chunking winner, reranker gain, latency/cost trade-off, multi-hop, when
   GraphRAG/Agentic help).
5. **Learning documentation** → `docs/LEARNING.md` (concept encyclopedia),
   `README.md`, `PLAN.md`, plus per-experiment notes in the report.

## Execution order

1. Scaffold repo + GitHub remote.
2. Corpus (14 RAG-topic docs, cross-linked for multi-hop questions) + golden
   dataset (50 questions: 30 single-hop, 8 multi-hop, 6 tricky, 6 comparative).
3. Core modules (corpus → chunking → embeddings → stores).
4. Retrievers + rerankers.
5. Generation + LLM providers.
6. Eval metrics + golden tooling.
7. Graph, agents (phase 6).
8. Experiment runner: baseline → chunking matrix → retrieval matrix →
   challenges → phase-6 patterns; write `experiments/results.json`.
9. HTML dashboard generator → `site/index.html`.
10. Docs + final verification + push.

## Ground rules

- The golden dataset is the single source of truth for comparisons; every
  variant runs over the exact same questions.
- Offline-mode numbers are labeled as such in the report; online mode is one
  `cp .env.example .env` away.
- No secrets committed; `.env` is gitignored.
