# FINDINGS — Evaluation Report

Numbers come from `experiments/results.json`: all 16 variants over the same
51-question golden set (31 single-hop, 8 multi-hop, 6 tricky, 6 comparative)
on a 16-doc, ~8k-word corpus. **This primary run uses DeepSeek
(`deepseek-chat`) for generation, query transforms, critiques, and LLM-judge
metrics — 1,330 API calls, $0.21 total.** The offline-mode run (extractive
generator, zero API cost) is kept in `experiments/results-offline.json` for
comparison, and the dashboard (`site/index.html`) shows both side by side.

## The five deliverable questions

### 1. Which chunking strategy worked best and why?

**Answer: sentence-aware recursive chunking wins on ranking; the mechanism
is measurable, and the caveat is just as interesting as the win.**

| Chunker (same vector retriever) | Doc recall@5 | Hit | MRR | NDCG@5 | Correctness | Faithfulness (judge) |
|---|---|---|---|---|---|---|
| fixed (baseline) | 91.2 | 49.0 | 32.3 | 35.6 | 35.5 | 94.5 |
| recursive | 92.2 | 52.9 | **38.2** | **40.5** | 35.3 | 100.0 |
| semantic | 91.2 | 41.2 | 27.4 | 29.9 | 34.8 | 98.0 |
| parent-child | 88.2 | 41.2 | 31.8 | 34.2 | 33.2 | 92.5 |

Recursive chunking lifts MRR **+18%** and NDCG **+14%** over fixed windows
with the identical retriever. The **boundary-integrity diagnostic** shows
the mechanism: fixed windows cut **25.5% of corpus sentences** in half
(semantic: 0%, recursive: 24.5% — long sentences only, parent-child: 5%).
And the targeted analysis confirms it end-to-end: on the 27 questions whose
supporting sentences fixed chunking damages, sentence-aware chunking scores
higher; on questions fixed chunking happens to leave intact, the strategies
tie.

**The caveat is a finding in itself:** in live-LLM mode the
answer-correctness gaps between chunkers shrink (35.5 vs 35.3 vs 34.8),
because a strong generator *infers across damaged context* — it reads the
first half of a split sentence and reconstructs the rest. In offline
extractive mode the same gaps were much wider (17.3 vs 17.8, and the
damaged-subset gap 16.1→17.4). **Chunking buys most where the generator is
weakest** — extractive systems, small models, and every ranking metric, in
every mode.

### 2. How much did the reranker improve results?

**Answer: the cross-encoder reranker was the reliable Phase 3 win (+40%
MRR, +12% hit, +4% correctness), while HyDE flipped from disaster offline
to star online — and that flip is the whole story about HyDE.**

| Retriever (recursive chunks) | Doc recall@5 | Hit | MRR | NDCG@5 | Correctness |
|---|---|---|---|---|---|
| vector | 92.2 | 52.9 | 38.2 | 40.5 | 36.2 |
| bm25 | 78.4 | 35.3 | 27.8 | 27.7 | 31.6 |
| hybrid (RRF) | 92.2 | 45.1 | 37.0 | 36.8 | 35.7 |
| multi-query | 93.1 | 56.9 | 41.8 | 41.5 | 34.3 |
| hyde | 89.2 | **58.8** | **46.9** | **46.3** | **39.0** |
| vector + rerank | 92.2 | 54.9 | 45.3 | 45.8 | 36.3 |
| hybrid + rerank | 92.2 | 54.9 | 45.3 | 45.8 | 37.0 |

The reranker needs no index change and no LLM calls — its cost is ~1.3s of
CPU per query (the one retrieval stage that registers in the latency
budget).

**HyDE's identity crisis, measured:** with offline lexical "hypothetical
documents" it lost 28 points of recall (63.7); with DeepSeek writing real
hypothetical answers it produced the best MRR (46.9) and the best overall
correctness (39.0), including the best single-hop correctness (46.4).
**HyDE's value is the quality of the language model writing the
hypothetical document.** Its residual risk shows in the multi-hop column
(19.0, worst): when the guess heads for the wrong neighborhood, the damage
is real.

### 3. What was the latency / cost trade-off?

**Answer: retrieval is noise except for the CPU reranker; the LLM call
dominates; and the whole smart-technique menu costs pocket change.**

| Component (median, Intel Mac CPU) | ms |
|---|---|
| embed query (cached MiniLM) | 0.01 |
| vector search (~180 chunks) | 0.04 |
| BM25 search (pure Python) | 3.1 |
| hybrid retrieval | 5.6 |
| hybrid + cross-encoder rerank (50 candidates) | 1,374 |
| **generation (DeepSeek API)** | **1,166** |

The reranker and the LLM call cost the same order of magnitude on this
hardware — rerank on GPU or via an API would drop to milliseconds while
the LLM call stays. Per-variant API cost ran $0.007–$0.033; the entire
16-variant × 51-question run cost **$0.21** (1,330 calls). The tokens
table in the dashboard shows exactly which variants spend: agentic 270
calls, advanced-combo 204, baseline 51.

### 4. How well did it handle multi-hop questions?

**Answer: multi-hop is where the baseline halves its answer quality, and
where the "smart" techniques actually earn their keep.**

Baseline (DeepSeek): correctness 39.7 single-hop vs **19.6 multi-hop** —
retrieval (93.8 doc recall) isn't the problem; assembling the answer from
two documents is. The winners on multi-hop answer quality are exactly the
techniques designed for it:

- **agentic: 30.6 multi-hop correctness (+56% over baseline)** — despite
  lower doc recall (the router sometimes skipped retrieval), its
  re-retrieval loop gathers the second hop and its drafts improve,
- hybrid+rerank: 26.9, self-rag: 24.8,
- HyDE: 19.0 — the wrong-neighborhood risk is worst here.

So the honest rule from this corpus: **multi-hop needs either more
retrieval (agentic) or better selection (rerank), not better paraphrasing
(HyDE).**

### 5. When does GraphRAG or Agentic RAG actually help?

**Answer: agentic helps exactly on multi-hop and costs on everything else;
graph is a specialist whose day hasn't come on this corpus; self-RAG is the
safety net that pays.**

- **Agentic RAG** (overall 34.6 vs baseline 35.5): routed 12 of 51
  questions past retrieval entirely (some correctly — greetings-style —
  some overconfidently, e.g. questions the model *thought* it knew),
  used ~1 step on average, declined 12, and still produced the best
  multi-hop correctness (30.6 vs 19.6). It cost 5x the calls of the
  baseline ($0.032 vs $0.010). The pattern is textbook: **the loop pays
  only where single-shot provably fails.**
- **GraphRAG** (32.3 correctness): helped the entity-link multi-hop
  questions, diluted single-hop ranking — net flat. On a corpus of 16
  documents the graph has too few edges to walk; its value needs a bigger,
  more relational corpus.
- **Self-RAG** (36.7 correctness — 2nd best, best single-hop 40.6, best
  comparative 41.8): critiqued every draft, re-retrieved 23 times,
  declined 16 questions outright. The declines were *correct* behavior —
  e.g. the five-decisions list was split across 60-word chunks and the
  model refused to answer from half a list. **Honesty is the feature
  working.**

## Offline vs live: what changes when a real LLM shows up

| Metric (mean across 16 variants) | Offline (extractive) | DeepSeek |
|---|---|---|
| Answer correctness | 16.7 | **35.1** |
| Doc recall@5 | 88.2 | 88.7 |
| MRR | 34.7 | 36.3 |
| Faithfulness (judge) | — (trivial by construction) | 95.2 |

Three lessons from the comparison:

1. **Retrieval metrics barely move** (88.2 → 88.7): retrieval quality is a
   property of the retriever, not the generator. Evaluate it once.
2. **Answer quality doubles** with a real generator: extractive answers are
   capped by retrieval; a generative model synthesizes across chunks.
3. **Faithfulness is only measurable with a judge + a real generator.**
   Extractive answers are faithful by construction (1.0 everywhere,
   useless as a metric); the judge run shows where the live model drifts
   (e.g. 74.7 on the agentic variant — the loop's merged context invites
   unsupported claims).

## Phase 5 challenge lab (live run)

| Experiment | Result |
|---|---|
| Forced bad retrieval (worst-k chunks) | correctness **36.8 → 12.9** (−65%); judge faithfulness 69.8 → 0 |
| Boundary integrity | fixed 74.5% / recursive 75.5% / semantic 100% / parent-child 95% of corpus sentences intact |
| Chunk damage → quality | on the 27 damaged questions: fixed 31.4 vs recursive 32.7 (live LLM partially repairs; offline gap was wider) |
| Deleting 07-reranking.md | doc recall 16.7% on affected questions, 100% on controls |
| Adding a new doc | q51: unanswerable → fully retrievable |
| Cost | $0.21 total, per-variant table in dashboard |

## What I would do differently next time

1. **Bigger corpus first.** 8k words demonstrates every effect; 50k+
   would make the chunking and graph stories stark and give the agentic
   router real work to do.
2. **Tune the agentic router + critic.** The router was overconfident on 4
   corpus questions, and the critic's verdicts were harsher than its own
   scores (sup=1.0 with "re-retrieve"). A threshold on the scores instead
   of the verdict would cut the 16 self-RAG declines to the genuinely
   unanswerable cases.
3. **Reranker on GPU or via API.** The 1.3s CPU reranker is the only
   retrieval-stage latency; an API reranker would collapse it.
4. **Both models, same harness.** The DeepSeek run is committed; the same
   command with `--llm openai` drops an OpenAI column into the comparison.

## Reproducing

```bash
pip install -r requirements.txt -r requirements-optional.txt
python scripts/run_experiments.py --llm deepseek --judge      # primary run (~$0.21)
python scripts/run_experiments.py                              # offline comparison
python site/build_report.py                                    # regenerate the dashboard
```
