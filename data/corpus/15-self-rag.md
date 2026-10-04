# Corrective RAG and Self-RAG

## The core idea

Naive RAG never checks its work: if retrieval returned garbage, the LLM
still answers confidently from garbage. Corrective and Self-RAG add a
critique step that inspects either the retrieval or the draft answer and
triggers a repair action: re-retrieve, refine, or admit ignorance.

## Self-RAG

Self-RAG (self-reflective RAG) teaches the model to emit reflection tokens
as it works — *retrieve* (is retrieval needed?), *isrel* (is this chunk
relevant?), *issup* (is the claim supported?), *isuse* (is the answer
useful?). The model decides per step whether to retrieve, which chunks to
use, and whether its draft is good enough. In practice, frameworks
approximate the trained tokens with prompted critiques, which is what this
project does.

## Corrective RAG (CRAG)

CRAG adds a lightweight retrieval evaluator before generation: it grades the
retrieval as correct, ambiguous, or incorrect. Correct results pass through.
Ambiguous ones get refinement — knowledge refinement or query rewriting.
Incorrect ones trigger re-retrieval or an external fallback (like web
search). The key contribution is treating "wrong retrieval" as a normal,
detectable event instead of a silent failure.

## "I don't know" is a feature

A pipeline that admits when the corpus cannot answer is more trustworthy
than one that confabulates. Both Self-RAG and CRAG produce this behavior
when re-retrieval also fails. Golden sets should contain unanswerable
questions precisely to test it — this project's tricky subset does.

## What this project implements

A prompted critique loop: generate a draft from the retrieved context, have
the critic judge whether the context supports the draft, re-retrieve once
with a revised query if not, and answer "I don't know" (with the best
partial context it found) if support is still lacking. Offline mode uses a
deterministic lexical support check in place of the LLM critic.
