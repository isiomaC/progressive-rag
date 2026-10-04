#!/usr/bin/env python
"""Targeted re-evaluation of the agentic + self-rag variants (bug fixes
applied), merged into an existing results file. Also re-aggregates all
metrics with the None-tolerant aggregate (fixes faithfulness_llm averages
when the first question had no judge score).

Usage: python scripts/rerun_agents.py --llm deepseek experiments/results-deepseek.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragprog.agents import AgenticRAG, SelfRAG
from ragprog.chunking import RecursiveChunker
from ragprog.config import from_env
from ragprog.corpus import load_corpus
from ragprog.embeddings import build_embedder
from ragprog.eval_metrics import aggregate, group_by_type
from ragprog.experiments import _evaluate_variant
from ragprog.generation import make_generator
from ragprog.golden import load_golden, resolve_relevant_chunks
from ragprog.llm import get_llm
from ragprog.retrieval import HybridRetriever, RerankRetriever, ScoredChunk, build_reranker
from ragprog.stores import CorpusIndex


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="deepseek")
    parser.add_argument("results_file")
    args = parser.parse_args()

    cfg = from_env()
    cfg.ensure_dirs()
    cfg.llm_provider = args.llm
    llm = get_llm(cfg)
    if llm.is_offline:
        print("offline mode — nothing to re-run")
        return 1

    docs = load_corpus(cfg.corpus_dir)
    golden = load_golden(cfg.golden_yaml)
    embedder = build_embedder(cfg, [d.text for d in docs])
    generator = make_generator(llm)
    k = cfg.chunk["retrieval_k"]

    recursive = [c for d in docs for c in RecursiveChunker(cfg.chunk["recursive_size"], cfg.chunk["recursive_overlap"]).chunk(d)]
    idx_rec = CorpusIndex.build(recursive, embedder)
    rel_rec = resolve_relevant_chunks(golden, recursive)
    reranker = build_reranker(cfg, idx_rec)
    base = RerankRetriever(HybridRetriever(idx_rec), reranker)

    meta = {
        "llm_judge": True,
        "llm": llm.name,
    }

    results = json.loads(Path(args.results_file).read_text())

    for name, agent, title, desc, phase in [
        ("agentic", AgenticRAG(base, llm, generator, max_steps=2), "Agentic RAG",
         "Route → rewrite → retrieve → critique → stop or re-retrieve (max 2 steps).", "6-agentic"),
        ("self-rag", SelfRAG(base, llm, generator, max_rounds=2), "Corrective / Self-RAG",
         "Generate → critique context support → re-retrieve or answer 'I don't know'.", "6-selfrag"),
    ]:
        print(f"re-running {name} ({len(golden)} questions)...")
        records: list[dict] = []

        def make_adapter(a):
            def run(q, k_):
                result = a.run(q, k_)
                records.append({
                    "n_steps": result.n_steps,
                    "step_log": result.step_log,
                    "used_retrieval": result.used_retrieval,
                    "answer": result.answer.text,
                })
                scored = [ScoredChunk(c.id, c.doc_id, c.text, 0.0, a.name)
                          for c in result.final_chunks]
                return scored, result.answer
            return run

        variant = _evaluate_variant(
            name, phase, title, desc, make_adapter(agent),
            recursive, rel_rec, golden, generator, llm, k, meta)

        for item, rec in zip(golden, records):
            variant["per_question"][item.id].update(
                {"n_steps": rec["n_steps"], "step_log": rec["step_log"],
                 "used_retrieval": rec["used_retrieval"]})
        n_steps = [r["n_steps"] for r in records]
        variant["agent_stats"] = {
            "mean_steps": sum(n_steps) / len(n_steps) if n_steps else 0.0,
            "n_no_retrieval": sum(1 for r in records if not r["used_retrieval"]),
            "n_declined": sum(1 for item in golden
                              if "don't know" in variant["per_question"][item.id]["answer"].lower()),
        }
        for i, v in enumerate(results["variants"]):
            if v["name"] == name:
                results["variants"][i] = variant

    # re-aggregate every variant's metrics with the fixed aggregate()
    # (faithfulness_llm averages were zeroed when the first question had None)
    for v in results["variants"]:
        pq = v["per_question"]
        metas = [{"id": item["id"], "type": item["type"]} for item in results["golden"]]
        v["metrics"] = {
            "overall": aggregate(list(pq.values())),
            "by_type": group_by_type(metas, pq),
        }

    Path(args.results_file).write_text(json.dumps(results, indent=2))
    print(f"merged into {args.results_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
