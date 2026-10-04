# Vector Stores and Indexes

## What a vector store does

A vector store is a database that answers "which stored vectors are closest
to this query vector?" At its core it needs two operations: insert vectors
at index time and similarity search at query time.

## In-memory brute force (what this project uses)

For corpora under roughly ten thousand chunks, the best index is a numpy
matrix: store all chunk embeddings as rows, normalize them once, and at query
time compute one matrix-vector dot product. This is O(n) per query but the
constant is tiny — a few milliseconds for our corpus — and it is exact. The
vector store used in every experiment here is this design, deliberately, to
keep the comparison focused on the RAG techniques rather than on index
infrastructure.

## Approximate nearest neighbors (ANN)

At million-chunk scale, brute force stops being practical and indexes trade a
little accuracy for speed:

- **HNSW** — a proximity graph where search greedily walks towards closer
  neighbors. The default choice in most vector databases.
- **IVF** — clusters the corpus, then searches only the nearest clusters.
- **FAISS** — a library implementing these and more, with GPU support.

ANN indexes are a deployment concern, not a quality concern: at our scale
they would change latency by milliseconds and quality by nothing.

## Keyword-aware storage

A vector store cannot answer "find the chunk containing `MAX_RETRIES`". For
that you keep a separate lexical index — an inverted index or a BM25 model —
over the same chunk texts. This project maintains both indexes side by side
so hybrid retrieval can fuse their rankings. The two indexes are rebuilt
together from the same chunk registry.

## Updates and freshness

Adding a document means: chunk it with the same chunker, embed the new chunks
with the same model, and append rows to the store. Deleting or updating a
document means removing or replacing its chunks. Staleness is one of the
classic RAG failure modes: if the knowledge base changes and the index is not
rebuilt, answers quietly reference deleted facts. The knowledge-base update
experiment in Phase 5 of this project measures exactly this effect.
