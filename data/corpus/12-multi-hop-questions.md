# Multi-Hop Questions

## What multi-hop means

A multi-hop question needs facts from two or more chunks — usually in
different documents — combined in the answer. Example: "Which reranking
technique does the chunking document recommend pairing with parent-document
retrieval?" The first hop finds the recommendation; the second hop explains
it. Neither chunk alone contains the full answer.

## Why single-shot retrieval fails

A single retrieval step returns the top-k for one query. For a multi-hop
question, the top-k is dominated by chunks matching the most salient
keyword — usually hop one's topic — and the chunks for hop two never make
the cut. The generator then either guesses the missing hop (hallucination)
or answers only half the question. This is measurable: our evaluation
reports retrieval metrics separately for single-hop and multi-hop subsets,
and multi-hop recall@k drops sharply for every one-shot retriever.

## Strategies

- **Decomposition** — split the question into sub-questions, answer each
  with its own retrieval, combine. Costs extra LLM calls.
- **Iterative retrieval** — retrieve, read, and re-retrieve with a new query
  built from what was found. This is the agentic pattern.
- **Graph-guided retrieval** — traverse entity links between chunks instead
  of relying on a single similarity ranking. See the GraphRAG document.
- **Parent-document context** — a parent chunk often happens to contain both
  hops when they live in the same section; this recovers some multi-hop
  questions for free.

## Evaluation notes

Multi-hop evaluation needs per-hop ground truth: the golden entry lists all
documents that must be retrieved, and a multi-hop hit is scored only when
every required document appears in the context. Counting "any hop found" as
success flatters pipelines that answer half a question.
