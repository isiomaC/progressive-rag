# Progressive RAG Documentation Assistant

An exploratory, learning-first project that builds a question-answering system
over a documentation corpus and **progressively upgrades it** through five
generations of RAG:

```
Naive → Chunking Lab → Advanced Retrieval → GraphRAG → Agentic → Corrective/Self-RAG
```

Built to answer one question honestly: *which techniques actually move the
numbers, and at what cost?*

## Quick start

```bash
git clone git@github.com:isiomaC/progressive-rag.git
cd progressive-rag
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-optional.txt   # optional extras for real embeddings
python scripts/run_experiments.py          # runs the whole experiment matrix
python site/build_report.py                # regenerates site/index.html
open site/index.html                       # visual dashboard of every finding
```

No API key needed — the project runs in offline demo mode by default
(local embeddings + deterministic extractive answers). For real LLM answers,
HyDE, and LLM-judge metrics: `cp .env.example .env`, add `OPENAI_API_KEY`,
rerun.

## Repository map

| Path | What it is |
|---|---|
| `src/ragprog/` | the modular library (chunking, retrieval, eval, agents, graph) |
| `data/corpus/` | 14-document RAG documentation corpus (the knowledge base) |
| `data/golden/` | 50-question golden evaluation set with ground truth |
| `scripts/run_experiments.py` | runs Phases 1–6 and writes `experiments/results.json` |
| `site/` | self-contained HTML dashboard with charts + learning highlights |
| `docs/LEARNING.md` | concept encyclopedia: every technique explained |
| `docs/FINDINGS.md` | the final write-up with all the numbers |
| `docs/ARCHITECTURE.md` | how the components plug together |
| `PLAN.md` | the original implementation plan (phase mapping) |

## The headline findings

See `site/index.html` and `docs/FINDINGS.md` for full numbers. In short:

- **Chunking is the highest-leverage knob.** Fixed windows cut 25% of corpus
  sentences in half; sentence-aware recursive chunking lifts MRR +18% and
  NDCG +14% with the same retriever, and semantic chunking beats fixed on
  answer quality exactly on the questions fixed chunking damages.
- **Reranking buys the most per millisecond.** The cross-encoder reranker
  is the reliable Phase 3 win (+40% MRR, +7% correctness) — while HyDE lost
  28 points of recall when its guesses missed, and multi-query added
  nothing. Cost: ~1.3s CPU per query, everything else sub-10ms.
- **Multi-hop questions expose retrieval's ceiling** — correctness drops
  from 18.6 (single-hop) to 11.1 (multi-hop), and the composed/looping
  variants (HyDE, advanced combo, agentic) *hurt* multi-hop on this corpus:
  precision beats re-retrieval.
- **GraphRAG / agentic patterns are specialists, not upgrades.** Net flat
  to slightly negative here — they pay off on relational and multi-step
  workloads, which this small corpus mostly doesn't have. Measuring that
  honestly is the finding.
