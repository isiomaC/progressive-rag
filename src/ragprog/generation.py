"""Generation: prompt templates + generators (LLM online, extractive offline)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .chunking import Chunk
from .config import estimate_tokens
from .llm import LLM, Usage

RAG_PROMPT = """You are a documentation assistant. Answer the QUESTION using ONLY the CONTEXT below.
If the CONTEXT does not contain the answer, say "I don't know" — do not guess.
Cite the chunk numbers [n] that support each claim.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


@dataclass
class GenerationResult:
    text: str
    chunk_ids: list[str]
    usage: Usage
    latency_s: float


def format_context(chunks: list[Chunk]) -> str:
    parts = [f"[{i+1}] {c.text}" for i, c in enumerate(chunks)]
    return "\n\n".join(parts)


class Generator:
    """LLM-backed generation with the standard context/question prompt."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def generate(self, question: str, chunks: list[Chunk]) -> GenerationResult:
        t0 = time.perf_counter()
        context = format_context(chunks) if chunks else "(no context retrieved)"
        text, usage = self.llm.answer_with_context(question, context)
        return GenerationResult(
            text=text.strip(),
            chunk_ids=[c.id for c in chunks],
            usage=usage,
            latency_s=time.perf_counter() - t0,
        )


class ExtractiveGenerator:
    """Offline generator: builds the answer from the highest-scoring sentences
    in the retrieved context. Deterministic; used in offline demo mode."""

    def __init__(self, max_words: int = 120):
        self.max_words = max_words
        self.usage = Usage()

    def generate(self, question: str, chunks: list[Chunk]) -> GenerationResult:
        from .corpus import split_sentences
        from .eval_metrics import bigram_f1

        t0 = time.perf_counter()
        q_words = set(re.findall(r"[a-z0-9][a-z0-9-]*", question.lower()))

        candidates: list[tuple[float, int, str]] = []
        for ci, chunk in enumerate(chunks):
            for sent in split_sentences(chunk.text):
                if len(sent.split()) < 3:
                    continue
                score = bigram_f1(question, sent)
                if q_words:
                    score += 0.5 * len(q_words & set(re.findall(r"[a-z0-9][a-z0-9-]*", sent.lower()))) / len(q_words)
                candidates.append((score, ci + 1, sent))

        candidates.sort(key=lambda x: x[0], reverse=True)
        picked: list[str] = []
        used_chunks: list[int] = []
        n_words = 0
        for score, ci, sent in candidates:
            if score <= 0.0 or n_words >= self.max_words:
                break
            picked.append(sent)
            if ci not in used_chunks:
                used_chunks.append(ci)
            n_words += len(sent.split())

        if not picked:
            return GenerationResult(
                text="I don't know — the retrieved context does not contain an answer.",
                chunk_ids=[c.id for c in chunks],
                usage=self.usage,
                latency_s=time.perf_counter() - t0,
            )

        answer = " ".join(picked)
        cites = ",".join(f"[{i}]" for i in sorted(used_chunks))
        return GenerationResult(
            text=f"{answer} {cites}",
            chunk_ids=[c.id for c in chunks],
            usage=self.usage,
            latency_s=time.perf_counter() - t0,
        )


def make_generator(llm: LLM):
    """Extractive offline, LLM online — one interface."""
    return ExtractiveGenerator() if llm.is_offline else Generator(llm)
