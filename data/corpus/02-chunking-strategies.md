# Chunking Strategies

## Why chunking matters more than people think

Every downstream component sees the world through chunks. If the chunk that
contains the answer is malformed, no retriever can find it and no LLM can
read it. In practice, chunking quality often moves final answer quality more
than switching retrievers does. This is the single highest-leverage knob in a
RAG pipeline.

## Fixed-size chunking

Split text every N words (or tokens) with optional overlap. Simple and
predictable, but it slices through natural boundaries: a definition can start
at the end of one chunk and finish at the start of the next. With overlap the
reader still loses the middle. Fixed chunking is the baseline — fine for
uniform prose, weak on structured technical docs.

## Recursive character chunking

Recursive chunking splits on a hierarchy of separators — paragraphs first,
then sentences, then words — and only falls back to smaller separators when a
piece exceeds the target size. This keeps paragraphs and sentences intact far
more often than fixed chunking. Typical separator lists follow the pattern
`["\n\n", "\n", " ", ""]`. For documentation with headings, treating section
boundaries as the first separator preserves the author's structure.

## Semantic chunking

Semantic chunking groups sentences by meaning rather than length. A sliding
buffer accumulates sentences while their embedding stays similar to the
buffer's; when similarity between the buffer and the next sentence drops
below a threshold (we use the 90th percentile of breakpoints in the
document), the buffer becomes a chunk. The result is chunks that are variable
in length but topically coherent. It costs extra embedding calls at index
time, but not at query time.

## Parent-document (small-to-big) retrieval

Small-to-big splits the document twice: small "child" chunks for retrieval
and large "parent" chunks for the prompt. Production corpora commonly use
about 250-word children and about 1024-word parents; this lab scales down to
40-word children and 150-word parents because its corpus is small.
Retrieval over children is precise because small chunks match queries
tightly; generation over parents is complete because the model sees the
whole surrounding section. See the parent-document retrieval document for
the retrieval mechanics.

## Chunk-size trade-off

- **Small chunks** raise precision (better vector match, less noise in the
  prompt) but lose context and split answers.
- **Large chunks** preserve context but dilute the embedding signal and burn
  prompt tokens.
- **Overlap** mitigates boundary splits at the cost of duplicated tokens.

The practical rule of thumb for a documentation corpus: 200–500 words for
retrieval chunks, 500–1500 for context fed to the model. Chunk size must be
scaled to corpus size, though — this lab's corpus is ~8k words, so its
experiments use 60-word chunks, which is the same principle on a smaller
scale.

## A concrete failure of fixed chunking

Suppose the corpus contains the sentence: "the reranker re-scores the top
candidates with a cross-encoder, because joint encoding is far more
accurate than the bi-encoder similarity that produced the list." A
fixed-size window that ends after "cross-encoder," puts the word "because"
at the start of the next chunk and the explanation in a third. The user's
question "why is a cross-encoder more accurate?" matches the first chunk
well enough, but the chunk that actually answers *why* is two chunks away —
and a top-5 retriever that took the first chunk may never reach it. No
retriever upgrade can fix this: the information itself was cut in half.
This is the strongest argument that chunking is a bigger lever than
retrieval tricks.

## Section-aware chunking for docs

Documentation has its own natural unit: the heading plus everything under
it. Chunkers that respect markdown headings keep a definition together with
its explanation, which matters because documentation readers (and
evaluators) ask about concepts, not sentences. Recursive chunking with
`#`-level boundaries as the first separator approximates this; a fully
section-aware chunker treats each section as one logical chunk and only
subdivides oversized sections. This project's recursive chunker uses the
paragraph hierarchy for the same reason.

## Measuring chunk quality directly

Before running any end-to-end experiment, three cheap diagnostics reveal a
chunker's health:

1. **Boundary integrity** — how many known concept sentences (from the
   golden answers) are split across chunk boundaries? A chunker that splits
   golden sentences will lose those questions no matter what.
2. **Length distribution** — extremely short chunks (under ~10 words)
   dilute the embedding signal; extremely long ones dilute precision.
3. **Retrieval oracle** — with perfect retrieval (relevant chunks by
   construction), what is the ceiling on answer quality? The gap between
   this oracle and the real retriever is the retriever's fault; the gap
   between the oracle and perfect is the chunker's fault.
