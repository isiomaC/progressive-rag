# Embeddings and Vector Space

## What an embedding is

An embedding model maps text to a point in a high-dimensional vector space
such that semantically similar texts land close together. Similarity is
measured with cosine similarity (angle between normalized vectors) or dot
product.

## Common models

- **all-MiniLM-L6-v2** — a compact sentence-transformer. 384 dimensions,
  runs on CPU, good enough for documentation corpora. Used locally in this
  project by default.
- **text-embedding-3-small / 3-large (OpenAI)** — API embeddings, 1536+ dims.
  Strong, but every call costs money and leaves your machine.
- **bge-small / bge-large** — open bi-encoders that score near the top of
  retrieval leaderboards.

## Bi-encoders vs cross-encoders

Embedding models are bi-encoders: query and document are encoded
independently, so document vectors can be precomputed and stored. The price
is that the query and document never interact inside the model. A
cross-encoder reads the pair jointly, which makes it much more accurate at
judging relevance, but too slow to run over the whole corpus — which is why
cross-encoders are used as rerankers, not as indexers. See the reranking
document.

## Practical properties

- Embeddings are normalized to unit length in this project, so cosine
  similarity is a single dot product per candidate chunk.
- Brute-force search over a few thousand chunks is effectively free
  (milliseconds); approximate nearest-neighbor indexes only matter at much
  larger scale.
- The embedding space is only trustworthy for the same model and language:
  never mix query embeddings from one model with chunk embeddings from
  another.
- Embedding quality degrades for highly technical tokens (error codes,
  identifiers) — the keyword index exists precisely to cover that blind
  spot.

## Reading a similarity score

Cosine similarity runs from -1 to 1 but in practice, with a good bi-encoder,
almost all document pairs score between 0 and 0.8. Scores below 0.2 usually
mean unrelated text; scores above 0.6 usually mean strongly related text;
the wide middle band is where ranking errors live. This is why a reranker
matters: the retriever's 50 candidates all sit in the middle band, and the
cross-encoder's job is to separate the genuinely relevant ones.

Two practical consequences. First, thresholds must be calibrated per model
and per corpus — a 0.5 threshold from one embedding model means something
different in another. Second, low similarity is diagnostic: a query whose
best score is under the corpus's typical floor is a query the knowledge
base probably cannot answer, and flagging it before generation is the
cheapest form of bad-retrieval detection.

## Normalization and batching

Embeddings in this project are normalized to unit length at index time, so
query-time search is a single matrix-vector dot product with no distance
computation. Index-time embedding is batched (one model call for all
chunks) because the transformer's forward pass is dramatically faster per
text in batch than per text in a loop. Query-time embedding is a single
text and should be cached per (model, question) — this project memoizes
embeddings on disk so rerunning the experiment matrix never re-embeds the
same text twice.

## When embeddings fail

Three cases deserve a non-embedding fallback:

1. Exact identifiers — the string `MAX_RETRIES` embeds near `MAX_ATTEMPTS`
   and `retry limit`, which is fine for paraphrase but useless when the
   user needs the literal constant.
2. Numbers and versions — "2.7.0" and "2.7.1" embed nearly identically
   while the answer differs completely.
3. Out-of-domain text — an embedding model trained on general text will
   rank a chunk about basketball above a chunk about hashing if the corpus
   is mostly about storage systems.

All three argue for the hybrid index described in the BM25 document.

## Offline fallback

When sentence-transformers is unavailable, this project falls back to a
deterministic hashing-TFIDF embedder: n-gram hashes weighted by inverse
document frequency. It is a lexical, not semantic, embedding — results in
that mode are therefore a lower bound on what real embeddings achieve, and
the report labels them as such.
