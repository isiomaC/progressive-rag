# Introduction to Retrieval-Augmented Generation

## What RAG is

Retrieval-Augmented Generation (RAG) is a pattern that gives a language model
access to an external knowledge base at generation time. Instead of relying
only on what the model memorized during training, a RAG pipeline fetches
relevant documents and inserts them into the prompt as context.

A naive RAG pipeline has five components, executed in order:

1. **Loader** — reads source documents (markdown, PDF, HTML) into text.
2. **Chunker** — splits the text into fixed-size pieces (chunks) with some
   overlap, for example 500 tokens with 50 tokens of overlap.
3. **Embedder** — converts each chunk into a vector using an embedding model.
4. **Index** — stores the vectors (often a vector database).
5. **Retriever + Generator** — at query time, embed the question, find the
   top-k most similar chunks, and feed them to the LLM with a prompt that
   separates context from the question.

## Why RAG exists

RAG solves three real problems with LLMs used alone:

- **Hallucination.** A grounded model can cite passages it actually read,
  which reduces (but does not eliminate) invented facts.
- **Stale knowledge.** The model's training cutoff cannot know about new
  documentation, new releases, or private notes. RAG reads them at query time.
- **Attribution.** Returning the chunks alongside the answer gives users a
  way to verify claims — the chunks double as citations.

RAG is usually the right first tool compared to fine-tuning: it is cheap,
immediate, updateable, and works with any model. Fine-tuning changes the
weights; RAG changes the prompt. They are complements, not rivals.

## The baseline pipeline (Phase 1 of this project)

Our baseline is deliberately naive so later improvements have something to
beat:

- fixed-size chunking with overlap — 100 words with 20-word overlap in this
  lab, scaled down from the 500/50 production norm because our corpus is
  only about ten thousand words (chunk size must be scaled to corpus size),
- local MiniLM embeddings (384 dimensions) into an in-memory cosine index,
- top-5 vector retrieval with no reranking,
- a prompt template with clearly separated CONTEXT and QUESTION sections.

Measured against the 50-question golden set, this baseline is the reference
point for every later phase. All experiments in this repository report
relative changes against it.

## Known weaknesses of the naive pipeline

- Fixed chunking cuts ideas in half; a question answered by the boundary of
  two chunks may be missed entirely.
- Pure vector search is weak on exact identifiers (function names, error
  codes, configuration keys) — a keyword index does better there.
- Top-k with no reranking puts irrelevant-but-similar chunks into the prompt,
  wasting context and degrading answers.
- A single retrieval step cannot answer multi-hop questions that need facts
  from two different documents.
- The pipeline has no notion of "I could not find this" — it answers anyway.

Each weakness maps to a later phase of this project: chunking experiments,
hybrid search, reranking, agentic loops, and self-critique.
