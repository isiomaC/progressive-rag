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
python3.12 -m venv .venv && source .venv/bin/activate   # Intel Mac: 3.12 needed for torch
pip install -r requirements.txt -r requirements-optional.txt
python scripts/run_experiments.py          # runs the whole experiment matrix (offline)
python site/build_report.py                # regenerates site/index.html
open site/index.html                       # visual dashboard of every finding
```

No API key needed — offline demo mode uses local embeddings + deterministic
extractive answers. For real LLM answers, HyDE, multi-query, agent
critiques, and LLM-judge metrics:

```bash
cp .env.example .env            # add DEEPSEEK_API_KEY and/or OPENAI_API_KEY
python scripts/run_experiments.py --llm deepseek --judge --out experiments/results-deepseek.json
python scripts/run_experiments.py --llm openai --judge --out experiments/results-openai.json
python site/build_report.py     # dashboard now includes the model-comparison section
```

## Repository map

| Path | What it is |
|---|---|
| `src/ragprog/` | the modular library (chunking, retrieval, eval, agents, graph) |
| `data/corpus/` | 16-document RAG documentation corpus (the knowledge base) |
| `data/golden/` | 51-question golden evaluation set with ground truth |
| `scripts/run_experiments.py` | runs Phases 1–6 and writes `experiments/results*.json` |
| `experiments/results.json` | primary run (DeepSeek, live LLM) — the dashboard's source |
| `experiments/results-offline.json` | offline demo-mode run, for the model comparison |
| `site/` | self-contained HTML dashboard with charts + learning highlights |
| `docs/LEARNING.md` | concept encyclopedia: every technique explained |
| `docs/FINDINGS.md` | the final write-up with all the numbers |
| `docs/ARCHITECTURE.md` | how the components plug together |
| `PLAN.md` | the original implementation plan (phase mapping) |

## The headline findings

See `site/index.html` and `docs/FINDINGS.md` for full numbers. Primary run:
DeepSeek (`deepseek-chat`) generation + LLM judges, 1,330 calls, $0.21 total.

- **Chunking shows in ranking, not just answers.** Sentence-aware recursive
  chunking lifts MRR +18% / NDCG +14% over fixed windows with the same
  retriever; the boundary-integrity diagnostic shows fixed windows cut 25%
  of corpus sentences in half. Caveat found: a strong LLM partially
  *infers across* split chunks — chunking buys most where the generator is
  weakest.
- **Reranking is the reliable win; HyDE is a quality test.** Cross-encoder
  rerank: +40% MRR, no index change, ~1.3s CPU. HyDE lost 28 points of
  recall with offline pseudo-docs and won the whole table (+best MRR 46.9,
  best correctness 39.0) with DeepSeek writing real hypothetical answers —
  HyDE's value is the LLM writing the guess.
- **Multi-hop halves baseline correctness (39.7 → 19.6)** — and agentic
  RAG's re-retrieval loop recovers it to 30.6 (+56%), exactly where the
  loop earns its 5x cost. On single-hop it just pays for nothing.
- **GraphRAG was net flat** (few edges on 16 docs); **Self-RAG declined 16
  questions honestly** — refusing to answer from split lists is the feature
  working. Offline vs live comparison: retrieval metrics barely move
  (88.2→88.7 recall), answer correctness doubles (16.7→35.1).
