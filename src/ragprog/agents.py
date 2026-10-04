"""Agentic RAG and Corrective/Self-RAG — Phases 6.3 and 6.4.

Both wrap a retriever in a decision loop; both return rich results so the
report can show *how many extra steps they took and whether it paid off*.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .chunking import Chunk
from .config import estimate_tokens
from .eval_metrics import bigram_f1
from .generation import ExtractiveGenerator, GenerationResult, Generator, format_context
from .llm import LLM, Usage
from .retrieval import Retriever, rrf_merge


@dataclass
class AgentResult:
    answer: GenerationResult
    n_steps: int
    used_retrieval: bool
    final_chunks: list[Chunk]
    step_log: list[str] = field(default_factory=list)


class AgenticRAG:
    """Decision loop: route → plan query → retrieve → judge sufficiency →
    stop or refine-and-retrieve again. Stop conditions: verdict ok, or
    max_steps exhausted."""

    name = "agentic"

    def __init__(self, retriever: Retriever, llm: LLM, generator, max_steps: int = 2):
        self.retriever, self.llm, self.generator, self.max_steps = retriever, llm, generator, max_steps
        self.usage = Usage()

    def run(self, question: str, k: int) -> AgentResult:
        t0 = time.perf_counter()
        chunks: list[Chunk] = []
        chunk_map: dict[str, Chunk] = {}

        if not self.llm.route(question):
            if self.llm.is_offline:
                result = self.generator.generate(question, [])
            else:
                # No context to be faithful TO — answer from general knowledge
                # instead of running the RAG prompt with an empty context
                # (which would force an "I don't know").
                text, usage = self.llm.answer_direct(question)
                result = GenerationResult(text=text, chunk_ids=[], usage=usage, latency_s=0.0)
            return AgentResult(result, 0, False, [], ["routed: no retrieval needed"])

        query = self.llm.rewrite(question)
        log = [f"query: {query!r}"]
        for step in range(self.max_steps):
            scored = self.retriever.retrieve(query, k)
            new_chunks = []
            for sc in scored:
                c = _to_chunk(sc)
                if c.id not in chunk_map:
                    chunk_map[c.id] = c
                    new_chunks.append(c)
            chunks.extend(new_chunks)
            if not chunks:
                break
            draft = self.generator.generate(question, chunks)
            critique = self.llm.critique(question, format_context(chunks), draft.text)
            log.append(
                f"step {step + 1}: retrieved {len(new_chunks)} new chunks, "
                f"critique={critique.get('verdict')} "
                f"(rel={critique.get('relevant', 0):.2f} sup={critique.get('supported', 0):.2f})"
            )
            if critique.get("verdict") == "ok":
                self.usage.add(draft.usage)
                return AgentResult(draft, step + 1, True, chunks, log)
            query = self.llm.refine_query(question, format_context(chunks))

        if not chunks:
            result = self.generator.generate(question, [])
        else:
            result = self.generator.generate(question, chunks)
        self.usage.add(result.usage)
        return AgentResult(result, self.max_steps, True, chunks, log + ["stopped: max steps"])


class SelfRAG:
    """Corrective loop: generate → critique context support → re-retrieve
    once with a refined query if unsupported → regenerate → if still
    unsupported, answer honestly ("I don't know") with the best context."""

    name = "self-rag"

    def __init__(self, retriever: Retriever, llm: LLM, generator, max_rounds: int = 2):
        self.retriever, self.llm, self.generator, self.max_rounds = retriever, llm, generator, max_rounds
        self.usage = Usage()

    def run(self, question: str, k: int) -> AgentResult:
        chunks: list[Chunk] = []
        chunk_map: dict[str, Chunk] = {}
        log: list[str] = []
        query = question

        for round_no in range(self.max_rounds):
            scored = self.retriever.retrieve(query, k)
            for sc in scored:
                c = _to_chunk(sc)
                if c.id not in chunk_map:
                    chunk_map[c.id] = c
                    chunks.append(c)
            if not chunks:
                return AgentResult(
                    self.generator.generate(question, []), round_no, True, [], log + ["no chunks found"])
            context = format_context(chunks)
            draft = self.generator.generate(question, chunks)
            critique = self.llm.critique(question, context, draft.text)
            log.append(
                f"round {round_no + 1}: verdict={critique.get('verdict')} "
                f"(sup={critique.get('supported', 0):.2f})"
            )
            if critique.get("verdict") == "ok":
                self.usage.add(draft.usage)
                return AgentResult(draft, round_no + 1, True, chunks, log)
            if critique.get("verdict") == "re-retrieve":
                query = self.llm.refine_query(question, context)

        # exhausted: honest answer with best partial context
        honest = GenerationResult(
            text="I don't know — the retrieved context does not support a confident answer. "
                 "Best partial context found, but I will not guess.",
            chunk_ids=[c.id for c in chunks],
            usage=Usage(),
            latency_s=0.0,
        )
        log.append("gave up: answered 'I don't know' rather than hallucinate")
        return AgentResult(honest, self.max_rounds, True, chunks, log)


def _to_chunk(sc) -> Chunk:
    return Chunk(id=sc.id, doc_id=sc.doc_id, text=sc.text, start=0, end=len(sc.text),
                 parent_id=sc.parent_id)
