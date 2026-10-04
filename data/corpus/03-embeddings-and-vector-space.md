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

## Offline fallback

When sentence-transformers is unavailable, this project falls back to a
deterministic hashing-TFIDF embedder: n-gram hashes weighted by inverse
document frequency. It is a lexical, not semantic, embedding — results in
offline demo mode are therefore a lower bound on what real embeddings
achieve.
