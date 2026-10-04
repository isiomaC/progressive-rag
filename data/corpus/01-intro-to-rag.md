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

- fixed-size chunking with overlap — 60 words with 10-word overlap in this
  lab, scaled down from the 500/50 production norm because our corpus is
  only about eight thousand words (chunk size must be scaled to corpus
  size),
- local MiniLM embeddings (384 dimensions) into an in-memory cosine index,
- top-5 vector retrieval with no reranking,
- a prompt template with clearly separated CONTEXT and QUESTION sections.

Measured against the 50-question golden set, this baseline is the reference
point for every later phase. All experiments in this repository report
relative changes against it.

## A worked example

Consider the question "why did my answer cite chunk 14 when I asked about
chunking?" In a naive pipeline the user has no way to know whether chunk 14
was even retrieved by similarity or hallucinated by the model. RAG makes the
process inspectable: every answer is traceable to a numbered chunk, and the
chunk numbers in the answer must exist in the retrieved context. That
traceability is the whole point of the CONTEXT/QUESTION prompt format — the
boundary between what the system read and what the user asked must be
unambiguous.

## The five failure modes, restated

Naive RAG fails in five observable ways, and each maps to one phase of this
project:

1. The chunk containing the answer is split across two chunks → chunking
   experiments fix this.
2. The question uses an exact identifier (function name, error code) that
   the embedding model blurs → hybrid search fixes this.
3. The top-k is full of similar-but-useless chunks → reranking fixes this.
4. The question needs two documents and one retrieval pass cannot reach
   both → agentic and graph retrieval fix this.
5. The corpus has no answer but the pipeline answers anyway → Self-RAG
   fixes this by admitting ignorance.

An evaluator that measures only final answers hides all five; an evaluator
that measures retrieval and generation separately shows exactly which stage
broke. This project measures both, on every question, for every variant.

## RAG vs fine-tuning, concretely

Fine-tuning rewrites the weights; RAG rewrites the prompt. When a document
changes, a fine-tuned model needs a new training run, while a RAG index
needs only that document re-embedded. When a fact is wrong, a fine-tuned
model is wrong silently and identically for everyone; a RAG pipeline can
show the offending chunk and be corrected by fixing the document. The
projects that need fine-tuning are those where the *behavior* must change,
not the knowledge — style, format, tool use. The projects that need RAG are
those where knowledge changes faster than training runs.
