"""Evaluation metrics — Phase 4.

Retrieval metrics (no LLM needed, against golden relevant chunks/docs):
    hit_rate, precision@k, recall@k, mrr, ndcg@k,
    context_precision (RAGAS-style position-weighted), context_recall.

Generation metrics:
    faithfulness (lexical proxy for the RAGAS LLM judge),
    answer_relevancy, answer_correctness (token F1 vs ground truth),
    plus optional LLM-judge versions when online.

All lexical helpers use bigram overlap — deterministic and free, and the
numbers are labeled as proxies in the report.
"""

from __future__ import annotations

import math
import re

from .corpus import split_sentences

WORDS_RE = re.compile(r"[a-z0-9][a-z0-9-]*")


def _tokens(text: str) -> list[str]:
    return WORDS_RE.findall(text.lower())


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return {(a, b) for a, b in zip(tokens, tokens[1:])}


def bigram_f1(a: str, b: str) -> float:
    ga, gb = _bigrams(_tokens(a)), _bigrams(_tokens(b))
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    p, r = inter / len(ga), inter / len(gb)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def bigram_coverage(a: str, b: str) -> float:
    """One-directional: what fraction of a's bigrams appear in b?
    The right shape for faithfulness — an answer sentence is 'supported'
    when most of ITS phrasing appears in the context, not when the context's
    huge vocabulary happens to echo the sentence (F1's recall term would
    drown that signal)."""
    ga, gb = _bigrams(_tokens(a)), _bigrams(_tokens(b))
    if not ga:
        return 0.0
    return len(ga & gb) / len(ga)


def support_fraction(answer: str, context: str) -> float:
    """Faithfulness proxy: fraction of answer sentences with lexical support
    in the context (bigram coverage ≥ 0.3)."""
    sentences = [s for s in split_sentences(answer) if len(s.split()) >= 4]
    if not sentences:
        return 0.0
    supported = sum(1 for s in sentences if bigram_coverage(s, context) >= 0.3)
    return supported / len(sentences)


def answer_relevancy(question: str, answer: str) -> float:
    """Proxy: lexical overlap between question and answer."""
    return bigram_f1(question, answer)


def answer_correctness(answer: str, reference: str) -> float:
    """Token-overlap F1 against the golden answer (RAGAS correctness uses
    overlap + an LLM judge; this is the deterministic part)."""
    ta, tr = set(_tokens(answer)), set(_tokens(reference))
    if not ta or not tr:
        return 0.0
    inter = len(ta & tr)
    p, r = inter / len(ta), inter / len(tr)
    return 2 * p * r / (p + r) if (p + r) else 0.0


# ---------------- retrieval metrics ----------------

def _retrieval_metrics(
    retrieved_ids: list[str],
    retrieved_doc_ids: list[str],
    relevant_chunk_ids: set[str],
    relevant_doc_ids: set[str],
    k: int,
) -> dict:
    retrieved_ids = retrieved_ids[:k]
    docs = retrieved_doc_ids[:k]
    rel = relevant_chunk_ids
    rel_docs = relevant_doc_ids

    hits = [i for i, cid in enumerate(retrieved_ids) if cid in rel]
    hit = 1.0 if hits else 0.0
    precision = len(set(retrieved_ids) & rel) / k if retrieved_ids else 0.0
    recall = len(set(retrieved_ids) & rel) / len(rel) if rel else 0.0
    mrr = 1.0 / (hits[0] + 1) if hits else 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(retrieved_ids) if cid in rel)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0

    # RAGAS-style context precision: precision@i averaged over relevant ranks
    cprec = 0.0
    if rel:
        n_seen = 0
        for i, cid in enumerate(retrieved_ids, start=1):
            if cid in rel:
                n_seen += 1
                cprec += n_seen / i
        cprec = cprec / len(rel)

    doc_hits = [i for i, d in enumerate(docs) if d in rel_docs]
    doc_recall = len(set(docs) & rel_docs) / len(rel_docs) if rel_docs else 0.0
    doc_hit = 1.0 if doc_hits else 0.0
    # multi-hop: ALL required docs present in the context window
    all_docs = set(docs)
    multi_hop_complete = 1.0 if (rel_docs and rel_docs <= all_docs) else 0.0

    return dict(
        hit_rate=hit,
        precision=precision,
        recall=recall,
        mrr=mrr,
        ndcg=ndcg,
        context_precision=cprec,
        context_recall=doc_recall,
        doc_hit=doc_hit,
        doc_recall=doc_recall,
        multi_hop_complete=multi_hop_complete,
    )


def evaluate_question(
    retrieved_ids: list[str],
    retrieved_doc_ids: list[str],
    relevant_chunk_ids: set[str],
    relevant_doc_ids: set[str],
    k: int,
    answer_text: str,
    reference_answer: str,
    question: str,
    context_text: str,
    llm=None,
) -> dict:
    r = _retrieval_metrics(retrieved_ids, retrieved_doc_ids, relevant_chunk_ids, relevant_doc_ids, k)
    g = dict(
        faithfulness=support_fraction(answer_text, context_text),
        answer_relevancy=answer_relevancy(question, answer_text),
        answer_correctness=answer_correctness(answer_text, reference_answer),
    )
    if llm is not None and not llm.is_offline:
        try:
            judge = llm.judge_support(context_text[:4000], answer_text[:2000])
            g["faithfulness_llm"] = float(judge.get("score", 0.0))
            g["correctness_llm"] = llm.judge_correctness(question, answer_text[:2000], reference_answer[:2000])
        except Exception:
            pass
    return {**r, **g}


def aggregate(per_question: list[dict], keys: list[str] | None = None) -> dict:
    """Mean over questions for numeric keys only; empty list → zeros."""
    if not per_question:
        return {}
    keys = keys or [k for k in per_question[0]]
    numeric = [k for k in keys if isinstance(per_question[0].get(k), (int, float))]
    return {k: sum(q.get(k, 0.0) for q in per_question) / len(per_question) for k in numeric}


def group_by_type(questions_meta: list[dict], metrics: dict[str, dict]) -> dict[str, dict]:
    """metrics: qid -> metric dict. Returns {type: aggregated}."""
    buckets: dict[str, list[dict]] = {}
    for meta in questions_meta:
        buckets.setdefault(meta["type"], []).append(metrics[meta["id"]])
    return {t: aggregate(vals) for t, vals in buckets.items()}


def detect_bad_retrieval(similarity_scores: list[float], threshold: float = 0.25) -> bool:
    """Phase 5 bad-retrieval detector: flag when the best chunk is weak."""
    return bool(similarity_scores) and max(similarity_scores) < threshold
