# FINDINGS — Evaluation Report

Numbers come from `experiments/results.json`: all 16 variants run over the
same 51-question golden set (31 single-hop, 8 multi-hop, 6 tricky, 6
comparative) on a 16-doc, ~8k-word corpus. The interactive dashboard
(charts + question-level heatmap) is `site/index.html`.

> **Mode note:** these numbers were produced in **offline demo mode** (no
> API key): local MiniLM embeddings, a deterministic extractive generator,
> lexical query transforms. They are a *lower bound* on live-LLM behavior —
> rerun with `--llm deepseek` or `--llm openai` for the real thing; the
> harness and report work identically, and faithfulness only becomes
> discriminative then (extractive answers are faithful by construction).

## The five deliverable questions

### 1. Which chunking strategy worked best and why?

**Answer: sentence-aware chunking (recursive) for ranking, semantic for
information integrity — and the mechanism is measurable.**

| Chunker (same vector retriever) | Doc recall@5 | Hit rate | MRR | NDCG@5 | Correctness |
|---|---|---|---|---|---|
| fixed (baseline) | 91.2 | 49.0 | 32.3 | 35.6 | 17.3 |
| recursive | 92.2 | 52.9 | **38.2** | **40.5** | 16.9 |
| semantic | 91.2 | 41.2 | 27.4 | 29.9 | **17.8** |
| parent-child | 88.2 | 41.2 | 31.8 | 34.2 | 15.2 |

Recursive chunking lifts MRR **+18%** and NDCG **+14%** over fixed windows —
the cleanest chunking signal in the run. The mechanism is visible in the
**boundary-integrity diagnostic**: fixed windows cut **25.5% of corpus
sentences** in half; semantic chunking keeps 100% intact; recursive 75.5%;
parent-child 95%. The smoking gun is the targeted analysis: on the 27
questions whose supporting sentences are damaged by fixed windows, semantic
chunking beats fixed on answer correctness (17.4 vs 16.1); on questions
where fixed windows happen to land cleanly, the two tie (19.5 vs 19.3).

*Why differences aren't bigger:* the corpus is small and uniform, and the
offline extractive generator masks partial-context damage. With longer
answers and a real LLM, split-chunk damage would show up far more in answer
quality — the ranking numbers already do.

### 2. How much did the reranker improve results?

**Answer: the cross-encoder reranker was the most reliable Phase 3 win —
+40% MRR, +14% hit rate, +7% correctness — while HyDE and multi-query were
gambles that this corpus punished.**

| Retriever (recursive chunks) | Doc recall@5 | Hit | MRR | NDCG@5 | Correctness |
|---|---|---|---|---|---|
| vector | 92.2 | 52.9 | 38.2 | 40.5 | 16.9 |
| bm25 | 78.4 | 35.3 | 27.8 | 27.7 | 14.6 |
| hybrid (RRF) | 92.2 | 45.1 | 37.0 | 36.8 | 16.6 |
| multi-query | 92.2 | 51.0 | 33.9 | 36.8 | 16.2 |
| hyde | **63.7** | 31.4 | 26.6 | 26.3 | 15.1 |
| vector + rerank | 92.2 | 54.9 | 45.3 | 45.8 | 18.4 |
| **hybrid + rerank** | 92.2 | **54.9** | **45.3** | **45.8** | **18.7** |

The reranker changes rank order within the shortlist — no index change, no
LLM calls, only compute. HyDE, by contrast, *lost* 28 points of document
recall: the hypothetical-answer guesses embeds in the wrong neighborhood
and drags the retrieval with it. Multi-query helped nothing here because
its offline reformulations were near-duplicates. The general lesson: a
relevance model beats a language-model guess, reliably.

### 3. What was the latency / cost trade-off?

**Answer: everything except reranking is noise; the reranker is the one
retrieval cost that registers, and it's still cheap next to any LLM call.**

| Component (median, Intel Mac CPU) | ms |
|---|---|
| embed query (MiniLM, cached) | 0.01 |
| vector search (~180 chunks, numpy) | 0.04 |
| BM25 search (pure Python) | 3.1 |
| hybrid retrieval | 5.6 |
| **hybrid + cross-encoder rerank (50 candidates)** | **1,306** |
| generation (offline extractive) | 0.3 |

Two fun details: BM25 is slower than vector search at this scale (numpy
matrix math beats Python loops), and the reranker — ~1.3s on CPU — is the
entire retrieval-side latency budget. With a live LLM, generation would
dwarf it anyway (hundreds of ms to seconds), which is the real design rule:
optimize retrieval for *quality*, not speed.

*Cost:* every "smart" technique is a loan of tokens. Multi-query and HyDE
spend an LLM call before retrieval; reranking spends compute instead;
agentic loops spend both, per step. At gpt-4o-mini prices a base question
costs ~$0.00012 (≈420 input + 100 output tokens); a live run of this whole
suite would land in the tens of cents.

### 4. How well did it handle multi-hop questions?

**Answer: multi-hop is where single-shot retrieval shows its ceiling, and
where the strongest pipelines (rerank, self-RAG) hold on best.**

Baseline by question type: doc recall 96.8 single-hop vs 93.8 multi-hop;
answer correctness **18.6 single-hop vs 11.1 multi-hop** — the real gap is
in answer quality, not retrieval. On the stricter "all required docs in
context" metric: baseline 87.5%, hybrid+rerank 87.5%, self-RAG 87.5% —
but **HyDE 25.0%, advanced-combo 25.0%, agentic 50.0%**. The composed and
looping variants *hurt* multi-hop here: HyDE's wrong-neighborhood guesses
and the offline critique's refines misdirect retrieval. Honest reading:
on a small corpus, multi-hop needs *precise* retrieval more than *more*
retrieval — reranking beats re-retrieval.

### 5. When does GraphRAG or Agentic RAG actually help?

**Answer: on this corpus, honestly — they mostly don't, and measuring that
is the finding.**

- **GraphRAG** (0.912 doc recall, 16.0 correctness): helps the handful of
  questions where the second hop is an entity link, but its entity
  expansion dilutes single-hop ranking — net flat to slightly negative.
  It's a relational specialist; the vector ranking already handles
  ordinary questions.
- **Agentic RAG** (0.902 doc recall, 17.1 correctness): routed 0 questions
  past retrieval, used exactly 1 step on average, and its critique-driven
  re-retrieval cost 2 multi-hop questions. On a small corpus, the loop has
  nothing to earn its keep on. Its value is provable only when single-shot
  retrieval provably fails — i.e., on the harder questions a bigger corpus
  would supply.
- **Self-RAG** (18.7 correctness — tied best): the corrective pattern paid
  off in the one place it should — declining 1 question outright instead
  of hallucinating, while keeping the strong base retriever's numbers.

**The meta-lesson:** every advanced pattern fixes a specific failure mode.
The baseline tells you where your own leaks are. On this corpus the leaks
were selection noise (reranking fixed it) and wrong-neighborhood guesses
(HyDE made it worse) — not missing structure, which is what graphs and
agents fix.

## Phase 5 challenge lab

| Experiment | Result |
|---|---|
| Forced bad retrieval (worst-k chunks) | correctness **17.0 → 6.7** (−60%) |
| Boundary integrity | fixed 74.5% / recursive 75.5% / semantic 100% / parent-child 95% of corpus sentences intact |
| Chunk damage → quality | on 27 damaged questions: fixed 16.1 vs semantic 17.4; on clean questions: tied |
| Deleting 07-reranking.md | doc recall 16.7% on affected questions (q13/q14/q32), 100% on controls |
| Adding a new doc | q51: unanswerable → fully retrievable (recall 0 → 1) |
| Bad-retrieval detector | best-similarity + keyword-overlap check before generation |

## What I would do differently next time

1. **Bigger corpus first.** 8k words is enough to *demonstrate* effects;
   50k+ would make them stark. Multi-hop and agentic findings especially
   need documents that genuinely separate the hops.
2. **Live LLM from day one.** The extractive generator compresses answer
   differences (faithfulness reads 1.0 by construction); a real generator
   + LLM judges would widen every gap the report shows.
3. **Smarter offline query transforms.** Multi-query and HyDE deserve
   real reformulations; lexical stand-ins undersell them.
4. **Agentic loop with a real critique.** The offline lexical critique is
   the weakest part of the loop; with an LLM critic the re-retrieve step
   would actually earn its cost on multi-hop.

## Reproducing

```bash
pip install -r requirements.txt -r requirements-optional.txt
python scripts/run_experiments.py              # offline demo mode
python scripts/run_experiments.py --llm deepseek   # DEEPSEEK_API_KEY in .env
python scripts/run_experiments.py --llm openai --judge
python site/build_report.py                    # regenerate the dashboard
```
