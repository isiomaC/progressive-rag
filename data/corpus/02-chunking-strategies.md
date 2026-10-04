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
50-word children and 250-word parents because its corpus is small.
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
scaled to corpus size, though — this lab's corpus is ~10k words, so its
experiments use 100-word chunks, which is the same principle on a smaller
scale.

## Measuring chunking

The right way to compare strategies is end-to-end on a fixed question set:
keep the retriever and generator constant, change only the chunker, and
measure both retrieval quality (hit rate, recall@k) and answer quality
(correctness, faithfulness). That is exactly the Phase 2 experiment in this
repository.
