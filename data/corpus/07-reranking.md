# Reranking

## Why a second ranking stage exists

The embedding index retrieves the top-k by one global similarity score. The
top 5 of 50 candidates often contains noise: chunks that are similar to the
query but not actually answering it. Reranking re-scores only the candidates
with a stronger model, so the final prompt gets better chunks for the same
token budget. This is the two-stage retrieve-then-rerank pattern.

## Cross-encoder rerankers

A cross-encoder reads the (query, chunk) pair jointly through a transformer,
outputting a relevance score. Because query and chunk interact inside the
model, cross-encoders are far more accurate than the bi-encoder similarity
that produced the candidates. The cost is compute: every candidate pair must
be re-encoded at query time, so you only rerank the top 20–100 candidates,
never the corpus. On CPU, a small cross-encoder scores a pair in tens of
milliseconds; batches of 50 candidates are comfortably interactive.

Popular options: **bge-reranker** (open source, strong), the small
**ms-marco MiniLM** cross-encoder used in this project (cheap on CPU), and
**Cohere's Rerank** API (managed, easy, per-call cost).

## Typical gains

On documentation corpora, reranking the top candidates typically lifts hit
rate and recall@k by a double-digit percentage over the raw retriever, with
no change to the index and milliseconds of added latency. It is the highest
accuracy-per-millisecond upgrade in this project — more reliable than query
transformations, which depend on the LLM guessing well.

## Interaction with chunking

Reranking fixes *selection* errors, not *chunking* errors: if the answer was
split across chunk boundaries, no reranker can recover it. Reranking and good
chunking are complementary, which is why the strongest configuration in this
project pairs parent-document chunking with a cross-encoder reranker.

## Fallback lexical reranker

When the cross-encoder model is unavailable (offline mode), this project
falls back to a lexical reranker — a weighted blend of BM25 and vector
similarity. It recovers exact-token matches the vector score missed, but it
cannot judge paraphrase quality the way a cross-encoder can.

## Reranking under the microscope

What the cross-encoder actually changes is *rank order within the
shortlist*, not membership. The retriever decides which 50 chunks get a
chance; the reranker decides which 5 reach the prompt. This division of
labor has a practical consequence: reranking cannot compensate for a
retriever that never found the right neighborhood — a retrieval miss at
rank 200 stays a miss. The retrievers that matter are therefore the ones
with high *recall* over a wide window (hybrid), while reranking buys
*precision* on the final five.

## Choosing a reranker

- Local small cross-encoder (ms-marco MiniLM): free, CPU-friendly, tens of
  milliseconds per pair, no privacy concerns. This project's choice.
- bge-reranker: stronger, still local, heavier on CPU.
- API rerankers (Cohere): strongest and easiest, but every query leaves
  your machine and costs money per call.

The decision mirrors the embedding decision: local until the quality gap
is provable, API when the corpus outgrows CPU patience. In both cases the
swap is one class behind a small interface, which is exactly how this
project's rerankers are built.
