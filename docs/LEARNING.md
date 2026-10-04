# LEARNING.md — the concept encyclopedia

Every technique implemented in this project, explained the way I wish it had
been explained to me. Each entry says *what it is*, *why it works*, *what it
costs*, and *what our experiments showed*.

---

## 1. Naive RAG and its five parts

**What:** loader → chunker → embedder → index → retriever+generator.

**Why:** LLMs hallucinate, are stale, and can't cite. RAG fixes all three by
pasting retrieved context into the prompt at generation time.

**The key insight nobody tells you up front:** every later "advanced"
technique fixes a failure mode of *one specific component*. Naive RAG isn't
a bad idea — it's a checklist of where the leaks are.

| Weakness of naive RAG | Fixed by |
|---|---|
| chunks split answers in half | chunking strategies (Phase 2) |
| exact identifiers not found | BM25/hybrid (Phase 3) |
| top-k contains noise | reranking (Phase 3) |
| one query, one answer | multi-query / HyDE (Phase 3) |
| can't measure anything | eval framework (Phase 4) |
| answers from stale/missing docs | KB update discipline (Phase 5) |
| needs 2+ docs | graph / agentic (Phase 6) |
| never says "I don't know" | Self-RAG (Phase 6) |

---

## 2. Chunking — the highest-leverage knob

**Fixed size** — split every N words with overlap. Predictable; slices
through ideas.

**Recursive character** — try paragraph breaks first, then sentences, then
words; only fall back when a piece is oversized. Preserves the author's
structure. (LangChain's default approach.)

**Semantic** — split into sentences, embed them, walk with a buffer, break
when buffer↔next-sentence similarity drops below a threshold (we use the
90th percentile of a document's own breakpoints — self-calibrating).
Variable-length, topically coherent chunks. Extra embedding cost at index
time only.

**Parent-document (small-to-big)** — two granularities: children for
retrieval (precision), parents for the prompt (context completeness).
Production sizes ~250w children / ~1024w parents; we scaled to 50w/250w
because our corpus is ~10k words. **Chunk size must scale with corpus
size** — a 500-word chunk in a 10k-word corpus leaves you with 20 chunks
and nothing to retrieve.

**Why chunking beats retriever tricks:** if the answer is split across two
chunks, *no retriever can ever find it* — the information itself is
damaged. Rerankers fix *selection* errors; chunking fixes *information*
errors. Our experiments: semantic and parent-child beat fixed with the
identical retriever.

---

## 3. Embeddings & vector search

- Bi-encoders encode query and doc independently → docs precomputed and
  indexed; cheap, but the two texts never interact inside the model.
- all-MiniLM-L6-v2 (384-dim, local, CPU) is plenty for a docs corpus.
- Cosine over normalized vectors = one dot product. Brute force over a few
  thousand chunks is milliseconds — **ANN indexes are a scale problem, not
  a quality problem**.
- Never mix embeddings from different models: each defines its own space.
- Embedding models blur exact tokens (error codes, config keys) — that's
  the blind spot BM25 covers.

---

## 4. BM25 and hybrid retrieval

**BM25** = TF saturated by k1 (1.5), length-normalized by b (0.75), weighted
by IDF. Exact-token matcher: wins on identifiers, version numbers, config
keys.

**Hybrid** = run both indexes, merge rankings with **Reciprocal Rank
Fusion**: `score = Σ 1/(60 + rank)`. The trick: ranks are comparable across
heterogeneous indexes even when raw scores aren't. No calibration needed.

**When to prefer which:** paraphrase/topical questions → vector; exact-token
questions → BM25; real corpora mix both → hybrid. Hybrid's cost is
sub-millisecond (BM25) + one more index build.

---

## 5. Query transformation

**Query rewriting** — one LLM call to make the question self-contained.
Cheapest; adopt first.

**Multi-query** — N reformulations, N retrievals, RRF-merge. Covers
vocabulary mismatch; costs 1 LLM call + N passes.

**HyDE** — embed an LLM-written *hypothetical answer* instead of the
question. Works because the model guesses how the corpus phrases the answer
— the pseudo-doc is closer to real documents than the raw question is. Costs
1 LLM call; **can backfire when the guess is confidently wrong** (our HyDE
variant scored *below* baseline document recall — that's the risk made
visible).

**Step-back** — retrieve on a broader version of the question, answer the
specific one with both.

Budget order: rewrite → multi-query → HyDE (riskiest).

---

## 6. Reranking

Retrieve 50 cheaply, rescore the shortlist with a **cross-encoder** (query
and chunk interact inside the model), keep top-k. Far more accurate than the
bi-encoder similarity that made the candidates; too slow for the corpus, so
it only touches candidates.

- bge-reranker / ms-marco-MiniLM (local, CPU) / Cohere Rerank (API).
- Typical gain: double-digit lift in hit/recall, no index change.
- Cost: tens of ms to ~2s per query on CPU depending on candidate count.
- **Reranking fixes selection, not chunking** — the two compose.

---

## 7. Evaluation — measure both halves

**Retrieval** (no LLM needed): hit rate, precision/recall@k, MRR, NDCG@k,
RAGAS context precision/recall.

**Generation** (needs a judge): RAGAS triad — faithfulness (is every claim
supported by the context?), answer relevancy (does it answer the question?),
answer correctness (vs ground truth). Implemented here as deterministic
lexical proxies, with LLM judges activating automatically when a key is
present.

**Golden sets:** 50–100 questions, mixed single-hop/multi-hop/tricky,
hand-written answers + required docs. Watch for **contamination** — writing
questions in corpus vocabulary flatters lexical retrievers.

**The proxy detail that matters:** faithfulness must be *one-directional
coverage* (what fraction of the answer's phrasing appears in the context),
not symmetric F1 — F1's recall term drowns the signal against a large
context.

---

## 8. Latency, cost, and staleness

- The LLM dominates end-to-end latency (90%+). Retrieval is noise by
  comparison — optimize it for *quality*, not speed.
- Output tokens cost ~4x input tokens (gpt-4o-mini: $0.15 vs $0.60 per 1M).
- Every "smart" retrieval trick is a **loan of tokens**: query transforms
  spend LLM calls up front; reranking spends compute instead; agents spend
  both, per step.
- **Staleness:** a KB change without an index rebuild = answers that quietly
  cite deleted facts. Our experiment: deleting one doc dropped recall to 0%
  on the questions that needed it while controls were unaffected; adding a
  doc flipped an unanswerable question to answerable.

---

## 9. Multi-hop questions

Needs facts from 2+ chunks (often 2+ docs). Single-shot retrieval fails
because the top-k is dominated by the most salient keyword (hop one); hop
two's chunks never make the cut. Measurable: doc recall collapses from
single-hop to multi-hop in our baseline.

Fixes: decomposition, iterative (agentic) retrieval, graph traversal,
parent-document context (recovers multi-hop for free when both hops live in
one section).

Scoring rule: a multi-hop hit counts only when **every** required document
is in the context — "any hop found" flatters half-answerers.

---

## 10. GraphRAG

Two families:

- **KG index (LlamaIndex style):** entities + relations; at query time
  expand question entities to neighbors and their chunks. Lightweight.
- **Community-summary (Microsoft style):** partition the entity graph
  (Leiden), summarize each community with an LLM, answer corpus-wide
  questions from summaries. Powerful for *global* questions, expensive to
  index.

Ours: glossary + capitalized n-grams as entities, co-occurrence edges,
1-hop expansion, RRF-merge with vector. No LLM at index time.

**When it helps:** relational questions ("which techniques are paired
together?"), multi-hop where hop two is an entity link, corpus-level
questions. **When it doesn't:** ordinary single-hop — the vector ranking
already works, so the graph just adds latency.

---

## 11. Agentic RAG

A loop of decisions: route (retrieval needed?) → plan query → retrieve →
judge sufficiency → stop or continue. Every decision is an LLM call, so it's
strictly more expensive than one-shot RAG.

**The hard part is stopping.** Without explicit stop conditions (max steps,
answerability check, confidence threshold), agents loop. Our agent stops on
a passed critique or 2 steps.

**When it pays:** multi-hop/multi-part questions, workloads with a mix of
retrieval and no-retrieval queries (the router saves those tokens).

---

## 12. Corrective RAG & Self-RAG

- **CRAG:** grade the retrieval (correct / ambiguous / incorrect) *before*
  generation; refine or re-retrieve accordingly; web fallback for
  incorrect. Treats bad retrieval as a normal, detectable event.
- **Self-RAG:** reflection tokens — retrieve, isrel, issup, isuse — deciding
  per step whether to retrieve, use, and trust. In practice: prompted
  critiques, which is what we implement.

**"I don't know" is a feature.** A pipeline that declines beats one that
confabulates. Tricky golden questions exist to measure exactly this: naive
pipelines answer them confidently and wrongly; Self-RAG declines.

---

## 13. How to read the experiments

- Compare variants one variable at a time: chunking matrix keeps the
  retriever fixed; retrieval matrix keeps chunking fixed.
- Watch doc-level recall (strategy-independent) as the primary retrieval
  axis and answer correctness as the end-to-end axis.
- The question-level heatmap shows *which* questions break — bad retrieval
  is question-specific, not uniform.
- Offline-mode numbers are lower bounds (extractive generator, lexical
  transforms); rerun with a key for real LLM behavior.
