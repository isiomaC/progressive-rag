# Agent Instructions

## This repo

Progressive RAG learning project. Before running experiments or rebuilding
the report, read `PLAN.md` (phase mapping) and `docs/ARCHITECTURE.md`.

- **Runtime:** use `.venv` (Python 3.12 — Intel Mac: torch has no newer
  wheels; see `requirements-optional.txt`).
- **Experiments:** `python scripts/run_experiments.py` writes
  `experiments/results.json`. Never edit results.json by hand.
- **Report:** `python site/build_report.py` regenerates `site/index.html`
  from results.json; the page must stay self-contained (no CDN).
- **Docs:** `docs/LEARNING.md` is the concept encyclopedia; keep it in sync
  with any new technique added to `src/ragprog/`.
- **Corpus & golden set:** `data/corpus/*.md` and
  `data/golden/questions.yaml` are paired — a corpus change must update any
  golden answers it invalidates, or the suite will silently score wrong
  facts as wrong.
- **Secrets:** never commit `.env`; providers read keys from the
  environment only.
