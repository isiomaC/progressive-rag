# Latency and Cost

## Where the time goes

End-to-end latency is dominated by the LLM call; everything else is usually
noise:

- Embedding one query: single-digit milliseconds (local) or tens of
  milliseconds (API round trip).
- Vector search over a few thousand chunks: milliseconds.
- BM25 search: under a millisecond.
- Cross-encoder reranking of 50 candidates: tens of milliseconds on CPU.
- LLM generation: hundreds of milliseconds to seconds — 90%+ of the total.

Optimizing retrieval is about *quality*, not speed; the LLM is the only
component whose latency you can materially reduce (smaller model, streaming,
shorter outputs).

## Where the tokens (and money) go

Cost has three parts:

- **Embedding cost** — paid once per chunk at index time, once per query at
  query time. Tiny relative to generation.
- **Prompt cost** — paid per query, scales with number and size of chunks
  plus the instruction. Input tokens are cheap; keeping chunks small saves
  money directly.
- **Generation cost** — output tokens are typically priced several times
  higher than input tokens (e.g. 4x on gpt-4o-mini: $0.15 input vs $0.60
  output per million tokens). Short answers are cheap; verbose ones are not.

Every "smart" retrieval technique is really a loan of tokens: multi-query
and HyDE spend LLM calls up front to improve retrieval, and reranking spends
compute instead of tokens. A cost-tracking experiment in this project logs
tokens and estimated dollars per variant so the trade-off is explicit.

## Practical levers

- Batch embedding calls at index time.
- Cache embeddings per (model, text) pair — this project does.
- Reuse retrieved context across repeated similar questions where safe.
- Prefer local models (MiniLM, small cross-encoder) for the cheap stages;
  reserve the paid LLM for generation.

## Reading a latency budget

A useful way to see the whole pipeline is as a fixed budget: retrieval can
spend freely (it is cheap), generation is the one expensive line item, and
everything between is optional debt. Concrete numbers from this project,
measured on a laptop CPU:

- query embedding ≈ single-digit ms,
- vector and BM25 search ≈ milliseconds,
- cross-encoder reranking of 50 candidates ≈ 1–2 seconds — the only
  retrieval stage that registers,
- offline extractive generation ≈ 1 ms — replace with a live LLM and this
  becomes hundreds of ms to seconds, dwarfing everything.

Two design rules follow. First, never measure retrieval quality by wall
clock — optimize retrieval for *quality* and pay for it out of the
millisecond budget. Second, if a technique (reranking) is your latency
floor, reduce its input — rerank 20 candidates instead of 50 — before
replacing it.

## Cost per question, worked out

For a typical question in this project: 5 chunks × ~60 words ≈ 300 words
of context ≈ 420 input tokens, plus ~100 output tokens. At gpt-4o-mini
prices that is about $0.00012 per question before any query
transformations. Multi-query adds one LLM call plus more context; HyDE the
same; an agentic loop multiplies both by its step count. The money story
matches the latency story: generation dominates, and every "smart" stage is
a small, visible loan against it.
