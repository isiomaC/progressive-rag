"""Experiment orchestrator — runs every phase and writes results.json.

Variants (all evaluated on the same 51-question golden set):
  Phase 1  baseline                 fixed + vector + plain prompt
  Phase 2  chunk-{fixed,recursive,semantic,parent-child}
  Phase 3  retr-{bm25,hybrid,multi-query,hyde,rerank,hybrid+rerank}
  Phase 6  advanced-combo, graph, agentic, self-rag
  Phase 5  challenges (bad retrieval, latency, cost, kb updates, multi-hop)
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict

from .agents import AgenticRAG, SelfRAG
from .chunking import Chunk, FixedChunker, ParentChildChunker, RecursiveChunker, SemanticChunker
from .config import Config, estimate_tokens
from .corpus import Document, load_corpus
from .embeddings import build_embedder
from .eval_metrics import answer_correctness, evaluate_question
from .generation import make_generator
from .golden import load_golden, resolve_relevant_chunks
from .graph import GraphRetriever, KnowledgeGraph
from .llm import LLM, Usage, get_llm
from .retrieval import (
    AdvancedComboRetriever, BM25Retriever, HybridRetriever, HyDERetriever,
    MultiQueryRetriever, ParentDocRetriever, RerankRetriever, Retriever,
    VectorRetriever, build_reranker,
)
from .stores import CorpusIndex


def _chunks_for(chunker, docs: list[Document]) -> list[Chunk]:
    return [c for d in docs for c in chunker.chunk(d)]


def _split_parents_children(chunks: list[Chunk]) -> tuple[list[Chunk], list[Chunk]]:
    parents = [c for c in chunks if c.parent_id is None]
    children = [c for c in chunks if c.parent_id is not None]
    return parents, children


def _index_over(index, extra_chunks: list[Chunk]) -> CorpusIndex:
    """Attach chunks to an index's lookup table (for parent mapping)."""
    for c in extra_chunks:
        index.chunk_by_id[c.id] = c
    return index


def _usage_delta(after: Usage, before: Usage) -> Usage:
    return Usage(
        prompt_tokens=after.prompt_tokens - before.prompt_tokens,
        completion_tokens=after.completion_tokens - before.completion_tokens,
        n_calls=after.n_calls - before.n_calls,
    )


def _evaluate_variant(
    name: str,
    phase: str,
    title: str,
    description: str,
    retriever,                       # callable(question, k) -> list[ScoredChunk]
    chunks: list[Chunk],
    relevant: dict[str, set[str]],
    golden,
    generator,
    llm: LLM,
    k: int,
    meta: dict,
) -> dict:
    per_question: dict[str, dict] = {}
    questions_meta = [
        {"id": q.id, "type": q.type} for q in golden
    ]
    context_texts: dict[str, str] = {}
    usage_total = Usage()
    lat_retr: list[float] = []
    lat_gen: list[float] = []

    for item in golden:
        llm_usage_before = Usage(llm.usage.prompt_tokens, llm.usage.completion_tokens, llm.usage.n_calls)

        t0 = time.perf_counter()
        try:
            scored = retriever(item.question, k)
        except Exception as exc:
            scored = []
            print(f"[warn] {name} {item.id}: {type(exc).__name__}: {exc}")
        lat_retr.append((time.perf_counter() - t0) * 1000)

        got_chunks = [Chunk(id=s.id, doc_id=s.doc_id, text=s.text, start=0, end=len(s.text), parent_id=s.parent_id) for s in scored]
        context = "\n\n".join(c.text for c in got_chunks)
        t0 = time.perf_counter()
        result = generator.generate(item.question, got_chunks)
        lat_gen.append((time.perf_counter() - t0) * 1000)

        u = _usage_delta(llm.usage, llm_usage_before)
        usage_total.add(u)

        metrics = evaluate_question(
            retrieved_ids=[c.id for c in got_chunks],
            retrieved_doc_ids=[c.doc_id for c in got_chunks],
            relevant_chunk_ids=relevant[item.id],
            relevant_doc_ids=item.relevant_doc_ids,
            k=k,
            answer_text=result.text,
            reference_answer=item.answer,
            question=item.question,
            context_text=context,
            llm=llm if meta.get("llm_judge") else None,
        )
        per_question[item.id] = {
            **metrics,
            "retrieval_ms": lat_retr[-1],
            "generation_ms": lat_gen[-1],
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "n_llm_calls": u.n_calls,
            "answer": result.text,
            "retrieved_doc_ids": [c.doc_id for c in got_chunks],
            "retrieved_chunk_ids": [c.id for c in got_chunks],
        }

    from .eval_metrics import aggregate, group_by_type
    price = getattr(llm, "price", (0.0, 0.0))
    return {
        "name": name,
        "phase": phase,
        "title": title,
        "description": description,
        "k": k,
        "metrics": {
            "overall": aggregate(list(per_question.values())),
            "by_type": group_by_type(questions_meta, per_question),
        },
        "latency": {
            "retrieval_ms_median": statistics.median(lat_retr) if lat_retr else 0,
            "generation_ms_median": statistics.median(lat_gen) if lat_gen else 0,
            "total_ms_median": statistics.median([r + g for r, g in zip(lat_retr, lat_gen)]) if lat_retr else 0,
            "retrieval_ms_p95": _p95(lat_retr),
        },
        "cost": {
            **usage_total.to_dict(),
            "context_words_per_query": sum(len(c.text.split()) for q in golden for c in []) if False else None,
            "usd": usage_total.cost(price) if not llm.is_offline else 0.0,
            "llm": llm.name,
        },
        "per_question": per_question,
    }


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[int(0.95 * (len(s) - 1))]


def _retrieve_worst(index: CorpusIndex, question: str, k: int):
    """Phase 5: force the WORST chunks (lowest similarity) into the prompt."""
    vec = index.search_vector(question, len(index))
    worst = vec[-k:] if len(vec) >= k else vec
    from .retrieval import ScoredChunk
    out = []
    for cid, s in reversed(worst):
        c = index.get(cid)
        out.append(ScoredChunk(c.id, c.doc_id, c.text, s, "worst-k"))
    return out


def run_experiments(cfg: Config, llm: LLM | None = None, limit: int | None = None,
                    llm_judge: bool = False) -> dict:
    cfg.ensure_dirs()
    llm = llm or get_llm(cfg)
    if isinstance(llm, LLM) and llm.is_offline:
        llm_judge = False

    t_start = time.perf_counter()
    docs = load_corpus(cfg.corpus_dir)
    golden = load_golden(cfg.golden_yaml)
    if limit:
        golden = golden[:limit]

    embedder = build_embedder(cfg, [d.text for d in docs])
    generator = make_generator(llm)
    k = cfg.chunk["retrieval_k"]

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus_docs": len(docs),
        "corpus_words": sum(d.word_count for d in docs),
        "golden_questions": len(golden),
        "golden_by_type": {t: sum(1 for q in golden if q.type == t) for t in sorted({q.type for q in golden})},
        "embedder": embedder.name,
        "embedder_semantic": embedder.semantic,
        "llm": llm.name,
        "llm_offline": llm.is_offline,
        "llm_judge": llm_judge,
        "chunk_config": cfg.chunk,
        "k": k,
    }

    # ---- chunkers & indexes ----
    fixed = _chunks_for(FixedChunker(cfg.chunk["fixed_size"], cfg.chunk["fixed_overlap"]), docs)
    recursive = _chunks_for(RecursiveChunker(cfg.chunk["recursive_size"], cfg.chunk["recursive_overlap"]), docs)
    semantic = _chunks_for(SemanticChunker(embedder, cfg.chunk["semantic_percentile"]), docs)
    pc_all = _chunks_for(ParentChildChunker(cfg.chunk["child_size"], cfg.chunk["parent_size"]), docs)
    pc_parents, pc_children = _split_parents_children(pc_all)

    idx_fixed = CorpusIndex.build(fixed, embedder)
    idx_rec = CorpusIndex.build(recursive, embedder)
    idx_sem = CorpusIndex.build(semantic, embedder)
    idx_pc = _index_over(CorpusIndex.build(pc_children, embedder), pc_parents)

    rel_fixed = resolve_relevant_chunks(golden, fixed)
    rel_rec = resolve_relevant_chunks(golden, recursive)
    rel_sem = resolve_relevant_chunks(golden, semantic)
    rel_pc = resolve_relevant_chunks(golden, pc_parents)   # parents carry the answers

    if llm.is_offline:
        llm.set_index(idx_rec)

    reranker = build_reranker(cfg, idx_rec)

    variants: list[dict] = []
    variants.append(_evaluate_variant(
        "baseline", "1-naive", "Naive RAG (baseline)",
        "Fixed chunking + vector top-k + plain context/question prompt. The reference point.",
        VectorRetriever(idx_fixed).retrieve, fixed, rel_fixed, golden, generator, llm, k, meta))

    for name, idx, rel, title, desc, phase in [
        ("chunk-fixed", idx_fixed, rel_fixed, "Fixed-size chunking",
         "100-word windows, 20-word overlap. Splits through natural boundaries.", "2-chunking"),
        ("chunk-recursive", idx_rec, rel_rec, "Recursive character chunking",
         "Separator hierarchy (paragraphs → sentences → words). Preserves structure.", "2-chunking"),
        ("chunk-semantic", idx_sem, rel_sem, "Semantic chunking",
         "Sentence-buffer breaks when similarity drops below the 90th percentile.", "2-chunking"),
        ("chunk-parent-child", idx_pc, rel_pc, "Parent-document (small-to-big)",
         "50-word children for retrieval, 250-word parents for the prompt.", "2-chunking"),
    ]:
        if name == "chunk-parent-child":
            retriever = ParentDocRetriever(VectorRetriever(idx_pc), idx_pc).retrieve
            chunks_for_relevance = pc_parents
        else:
            retriever = VectorRetriever(idx).retrieve
            chunks_for_relevance = {name: None}
        variants.append(_evaluate_variant(
            name, phase, title, desc, retriever,
            {"chunk-fixed": fixed, "chunk-recursive": recursive,
             "chunk-semantic": semantic, "chunk-parent-child": pc_parents}[name],
            rel, golden, generator, llm, k, meta))

    for name, retriever, title, desc in [
        ("retr-bm25", BM25Retriever(idx_rec).retrieve, "BM25 (keyword) retrieval",
         "Okapi BM25 on recursive chunks. Exact tokens only — no semantics."),
        ("retr-hybrid", HybridRetriever(idx_rec).retrieve, "Hybrid BM25 + vector (RRF)",
         "Both rankings fused with reciprocal rank fusion (k=60)."),
        ("retr-multi-query", MultiQueryRetriever(VectorRetriever(idx_rec), llm).retrieve,
         "Multi-query retrieval", "LLM reformulations merged with RRF. +1 LLM call, n passes."),
        ("retr-hyde", HyDERetriever(idx_rec, llm).retrieve,
         "HyDE", "Embed an LLM-written hypothetical answer instead of the question. +1 LLM call."),
        ("retr-rerank", RerankRetriever(VectorRetriever(idx_rec), reranker).retrieve,
         "Vector + reranker", f"Top-{cfg.chunk['rerank_candidates']} candidates rescored by a {reranker.name} reranker."),
        ("retr-hybrid-rerank", RerankRetriever(HybridRetriever(idx_rec), reranker).retrieve,
         "Hybrid + reranker", "RRF candidates, then cross-encoder (or lexical) rerank."),
    ]:
        variants.append(_evaluate_variant(
            name, "3-retrieval", title, desc, retriever, recursive, rel_rec,
            golden, generator, llm, k, meta))

    # ---- Phase 6 ----
    variants.append(_evaluate_variant(
        "advanced-combo", "6-advanced", "Advanced RAG combo",
        "Rewrite + multi-query + HyDE + rerank + parent-document, composed on parent-child chunks.",
        AdvancedComboRetriever(HybridRetriever(idx_pc), reranker, llm, idx_pc).retrieve,
        pc_parents, rel_pc, golden, generator, llm, k, meta))

    kg = KnowledgeGraph.build(recursive)
    variants.append(_evaluate_variant(
        "graph", "6-graph", "GraphRAG",
        "Entity co-occurrence graph: question entities → 1-hop neighbors → chunks, RRF-merged with vector.",
        GraphRetriever(idx_rec, kg).retrieve, recursive, rel_rec, golden, generator, llm, k, meta))

    base_strong = RerankRetriever(HybridRetriever(idx_rec), reranker)

    agentic = AgenticRAG(base_strong, llm, generator, max_steps=2)
    selfrag = SelfRAG(base_strong, llm, generator, max_rounds=2)
    for name, agent, title, desc, phase in [
        ("agentic", agentic, "Agentic RAG",
         "Route → rewrite → retrieve → critique → stop or re-retrieve (max 2 steps).", "6-agentic"),
        ("self-rag", selfrag, "Corrective / Self-RAG",
         "Generate → critique context support → re-retrieve or answer 'I don't know'.", "6-selfrag"),
    ]:
        variants.append(_evaluate_variant(
            name, phase, title, desc,
            lambda q, k_, a=agent: _run_agent(a, q, k_),
            recursive, rel_rec, golden, generator, llm, k, meta))

    challenges = _run_challenges(cfg, llm, generator, docs, golden,
                                 idx_rec, idx_fixed, rel_rec, embedder, meta)

    results = {
        "meta": meta,
        "golden": [
            {"id": q.id, "type": q.type, "question": q.question,
             "answer": q.answer, "docs": q.docs}
            for q in golden
        ],
        "variants": variants,
        "challenges": challenges,
        "total_runtime_s": round(time.perf_counter() - t_start, 1),
    }
    cfg.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.results_file, "w") as f:
        json.dump(results, f, indent=2)
    return results


def _run_agent(agent, question: str, k: int):
    """Agents need their full run; map back to a scored-chunk list for the
    shared evaluator, and stash the step log in the variant extra."""
    result = agent.run(question, k)
    from .retrieval import ScoredChunk
    scored = [ScoredChunk(c.id, c.doc_id, c.text, 0.0, agent.name) for c in result.final_chunks]
    if not hasattr(_run_agent, "logs"):
        _run_agent.logs = {}
    _run_agent.logs[question] = {"n_steps": result.n_steps, "log": result.step_log,
                                 "used_retrieval": result.used_retrieval,
                                 "answer": result.answer.text}
    return scored


def _run_challenges(cfg, llm, generator, docs, golden, idx_rec, idx_fixed,
                    rel_rec, embedder, meta) -> dict:
    k = cfg.chunk["retrieval_k"]
    out: dict = {}

    # ---- 1. Bad retrieval: force worst chunks, watch answers degrade ----
    from .eval_metrics import bigram_f1, support_fraction, answer_correctness
    affected = [q for q in golden if q.type in ("single-hop", "multi-hop")]
    good_corr, bad_corr = [], []
    good_faith, bad_faith = [], []
    for q in affected:
        good = [idx_fixed.get(cid).text for cid, _ in idx_fixed.search_vector(q.question, k)]
        worst = [c.text for c in _retrieve_worst(idx_fixed, q.question, k)]
        g_answer = generator.generate(q.question, [Chunk(id=str(i), doc_id="x", text=t, start=0, end=0) for i, t in enumerate(good)])
        b_answer = generator.generate(q.question, [Chunk(id=str(i), doc_id="x", text=t, start=0, end=0) for i, t in enumerate(worst)])
        good_corr.append(answer_correctness(g_answer.text, q.answer))
        bad_corr.append(answer_correctness(b_answer.text, q.answer))
        good_faith.append(support_fraction(g_answer.text, "\n\n".join(good)))
        bad_faith.append(support_fraction(b_answer.text, "\n\n".join(worst)))
    out["bad_retrieval"] = {
        "n_questions": len(affected),
        "correctness_good_mean": _mean(good_corr),
        "correctness_bad_mean": _mean(bad_corr),
        "faithfulness_good_mean": _mean(good_faith),
        "faithfulness_bad_mean": _mean(bad_faith),
        "note": "Forced worst-k chunks degrade answer correctness/fidelity; "
                "retrieval quality sets the ceiling generation can reach.",
    }

    # ---- 2. Latency breakdown per component (baseline vs advanced) ----
    q = golden[0].question
    n = 20
    t_emb = _timeit(lambda: embedder.embed_query(q), n)
    t_vec = _timeit(lambda: idx_rec.search_vector(q, k), n)
    t_bm25 = _timeit(lambda: idx_rec.search_bm25(q, k), n)
    from .retrieval import HybridRetriever, RerankRetriever, build_reranker
    reranker = build_reranker(cfg, idx_rec)
    t_hybrid = _timeit(lambda: HybridRetriever(idx_rec).retrieve(q, k), n)
    t_rerank = _timeit(lambda: RerankRetriever(HybridRetriever(idx_rec), reranker).retrieve(q, k), n)
    t_gen = _timeit(lambda: generator.generate(q, [Chunk(id=str(i), doc_id="d", text=c.text, start=0, end=0) for i, c in enumerate(idx_rec.chunks[:k])]), n)
    out["latency"] = {
        "embed_query_ms": t_emb,
        "vector_search_ms": t_vec,
        "bm25_search_ms": t_bm25,
        "hybrid_retrieval_ms": t_hybrid,
        "hybrid_plus_rerank_ms": t_rerank,
        "generation_ms": t_gen,
        "note": "The LLM (or extractive generator offline) dominates end-to-end "
                "latency; retrieval stages are single-digit to tens of ms.",
    }

    # ---- 3. Cost accounting (tokens per variant; online only) ----
    out["cost"] = {
        "llm": llm.name,
        "note": "Token/cost totals per variant live in results.variants[].cost; "
                "offline mode spends zero API tokens (extractive generator).",
    }

    # ---- 4. Knowledge-base updates: delete & add documents ----
    remaining = [d for d in docs if d.id != "07-reranking"]
    idx_no07 = CorpusIndex.build([c for d in remaining for c in RecursiveChunker(100, 20).chunk(d)], embedder)
    rel_no07 = resolve_relevant_chunks([q for q in golden if "07-reranking" in q.docs],
                                       idx_no07.chunks)
    affected_qs = [q for q in golden if "07-reranking" in q.docs]
    control_qs = [q for q in golden if q.type == "single-hop" and q.id not in {x.id for x in affected_qs}][:5]
    doc_recall_affected, doc_recall_control = [], []
    for q in affected_qs:
        got = [idx_no07.get(cid).doc_id for cid, _ in idx_no07.search_vector(q.question, k)]
        doc_recall_affected.append(len(set(got) & q.relevant_doc_ids) / len(q.relevant_doc_ids))
    for q in control_qs:
        got = [idx_no07.get(cid).doc_id for cid, _ in idx_no07.search_vector(q.question, k)]
        doc_recall_control.append(len(set(got) & q.relevant_doc_ids) / len(q.relevant_doc_ids))
    new_doc_qs = [q for q in golden if q.id == "q51"]
    if new_doc_qs:
        new_doc_q = new_doc_qs[0]
        idx_no16 = CorpusIndex.build(
            [c for d in docs if d.id != "16-token-budgets-and-prompt-templates"
             for c in RecursiveChunker(100, 20).chunk(d)], embedder)
        got16 = [idx_no16.get(cid).doc_id for cid, _ in idx_no16.search_vector(new_doc_q.question, k)]
        got16_with = [idx_rec.get(cid).doc_id for cid, _ in idx_rec.search_vector(new_doc_q.question, k)]
        add_doc_stats = {
            "q51_doc_recall_without_new_doc": 1.0 if any(d == new_doc_q.docs[0] for d in got16) else 0.0,
            "q51_doc_recall_with_new_doc": 1.0 if any(d == new_doc_q.docs[0] for d in got16_with) else 0.0,
        }
    else:
        add_doc_stats = {"q51_doc_recall_without_new_doc": None, "q51_doc_recall_with_new_doc": None}
    out["kb_update"] = {
        "deleted_doc": "07-reranking",
        "affected_question_ids": [q.id for q in affected_qs],
        "doc_recall_affected_mean": _mean(doc_recall_affected),
        "doc_recall_control_mean": _mean(doc_recall_control),
        "added_doc": "16-token-budgets-and-prompt-templates",
        **add_doc_stats,
        "note": "Deleting a doc silently degrades recall on questions that need "
                "it; adding a doc makes previously unanswerable questions "
                "answerable. The index must be rebuilt on every KB change.",
    }

    # ---- 5. Multi-hop breakdown lives in by_type metrics; summarize here ----
    out["multi_hop"] = {
        "note": "Per-variant multi-hop metrics are in results.variants[].metrics.by_type.multi-hop. "
                "Single-shot retrievers drop sharply on multi-hop; parent-doc, graph and agents recover some.",
    }
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _timeit(fn, n: int) -> float:
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000
