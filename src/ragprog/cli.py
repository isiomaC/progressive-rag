"""CLI: run experiments, inspect results, quick smoke test."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="ragprog", description="Progressive RAG experiments")
    sub = parser.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="run the full experiment matrix")
    run.add_argument("--llm", choices=["offline", "openai", "deepseek"], default=None,
                     help="LLM provider (default: from RAGPROG_LLM env or offline)")
    run.add_argument("--limit", type=int, default=None,
                     help="limit to the first N golden questions (quick runs)")
    run.add_argument("--judge", action="store_true",
                     help="also run LLM-judge generation metrics (online only)")
    run.add_argument("--out", default=None, help="results file override")

    rep = sub.add_parser("report", help="rebuild site/index.html from results")
    smoke = sub.add_parser("smoke", help="one question through the baseline pipeline")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return 1

    from .config import from_env
    from .experiments import run_experiments
    from .llm import get_llm

    cfg = from_env()
    cfg.ensure_dirs()
    if getattr(args, "llm", None):
        cfg.llm_provider = args.llm
    if getattr(args, "out", None):
        from pathlib import Path
        cfg.results_file = Path(args.out)

    if args.cmd == "run":
        llm = get_llm(cfg)
        results = run_experiments(cfg, llm, limit=args.limit, llm_judge=args.judge)
        print(f"\nwrote {cfg.results_file} ({len(results['variants'])} variants, "
              f"{results['total_runtime_s']}s)")
        return 0

    if args.cmd == "report":
        from site_build import build_report   # noqa — see site/build_report.py
        build_report(cfg)
        return 0

    if args.cmd == "smoke":
        from .corpus import load_corpus
        from .chunking import FixedChunker
        from .embeddings import build_embedder
        from .generation import make_generator
        from .retrieval import VectorRetriever
        from .stores import CorpusIndex

        docs = load_corpus(cfg.corpus_dir)
        emb = build_embedder(cfg, [d.text for d in docs])
        chunks = [c for d in docs for c in FixedChunker(100, 20).chunk(d)]
        idx = CorpusIndex.build(chunks, emb)
        llm = get_llm(cfg)
        gen = make_generator(llm)
        q = "What are the five components of a naive RAG pipeline?"
        scored = VectorRetriever(idx).retrieve(q, cfg.chunk["retrieval_k"])
        from .chunking import Chunk
        result = gen.generate(q, [Chunk(id=s.id, doc_id=s.doc_id, text=s.text, start=0, end=0) for s in scored])
        print("Q:", q)
        print("Retrieved:", [s.id for s in scored])
        print("A:", result.text[:300])
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
