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

- **Chunking is the highest-leverage knob.** Semantic and parent-child chunking
  beat fixed-size by a wide margin on answer quality for the same retriever.
- **Reranking buys the most per millisecond** of any Phase 3 technique;
  HyDE and multi-query help on hard queries but cost LLM calls.
- **Multi-hop questions are where single-shot retrieval breaks** — parent-doc
  and graph-guided retrieval recover some of the gap.
- **GraphRAG / agentic patterns only pay off** on relational and multi-step
  questions; on single-hop questions they add latency for zero gain.
