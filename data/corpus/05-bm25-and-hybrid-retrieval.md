# BM25 and Hybrid Retrieval

## What BM25 is

BM25 (Best Matching 25) is the classic sparse retrieval formula. For a query
term, a document scores higher when the term is frequent in that document
(saturated by the k1 parameter, typically 1.5), when the term is rare across
the corpus (inverse document frequency), and when the document is short
relative to the average length (the b parameter, typically 0.75).

Unlike embeddings, BM25 matches exact tokens. It therefore wins on queries
containing identifiers, function names, error codes, version numbers, and
configuration keys — precisely the vocabulary that embedding models blur.

## Where each index wins

- **Vector search** wins on paraphrase and topical similarity: "how do I make
  the program remember things" matches chunks about memory and state.
- **BM25** wins on exact matches: `MAX_RETRIES` must match the literal token.
- Neither dominates on a real documentation corpus, which mixes both — which
  is the argument for hybrid.

## Hybrid retrieval with reciprocal rank fusion

Hybrid retrieval runs both indexes and merges their rankings. The standard
merge is Reciprocal Rank Fusion (RRF): each chunk scores the sum of
`1 / (k + rank)` across the two lists, with k = 60 being the conventional
constant. RRF needs no score calibration — ranks are comparable across
indexes even when raw scores are not. The merged list keeps the top results.

Hybrid typically beats either index alone on mixed corpora. Its cost is
modest: the BM25 search adds sub-millisecond latency, and both indexes are
built once.

## Lexical reranking fallback

When the cross-encoder reranker is unavailable, this project reranks with a
lexical scorer: a blend of normalized BM25 and vector similarity. It is
cheaper and weaker than a cross-encoder, but it still improves over raw
vector ranking by recovering exact-token matches.

## Why fusion needs ranks, not scores

Vector similarity is cosine (roughly -1 to 1). BM25 scores are unbounded
and grow with query length. Averaging the two directly is meaningless —
one index always drowns the other. Reciprocal rank fusion sidesteps the
problem entirely by discarding magnitudes and keeping only order: each
list contributes 1/(k + rank), and a chunk must simply appear near the top
of at least one list to survive. That is the entire trick, and it is why
RRF is the default merge everywhere in this project.

## Failure mode: fusion dilution

RRF has a known weakness worth measuring: it *includes*, it never
*excludes*. If a bad signal — a misleading query reformulation, a
wrong-neighborhood HyDE guess — appears in one of the fused lists, its
chunks still enter the merged ranking and can push genuinely relevant
chunks out of the top-k. This is exactly why composing too many
transformations before fusion can *lower* quality, and why the strongest
configurations fuse at most two high-quality lists and then rerank the
merged candidates with a real relevance model.
