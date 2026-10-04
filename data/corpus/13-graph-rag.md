# GraphRAG

## The idea

Classic RAG treats chunks as independent facts. GraphRAG adds structure: an
index of entities (people, tools, concepts, systems) and the relationships
between them, built from the corpus, and retrieves by walking the graph
instead of only matching vectors.

## Two families

- **Knowledge-graph index (LlamaIndex style)** — extract entities and
  relations (usually with an LLM), store triples, and at query time expand
  the question's entities to their neighbors and the chunks that mention
  them. Lightweight, integrates with vector search.
- **Community-summary GraphRAG (Microsoft style)** — build the entity graph,
  partition it into communities (Leiden), summarize each community with an
  LLM, and answer global questions from community summaries plus local
  questions from subgraphs. Powerful for corpus-wide questions ("what are
  the main themes across these documents?") but expensive: entity extraction
  and summarization over the whole corpus.

## What this project implements

A simplified LlamaIndex-style graph: an entity glossary plus capitalized
multi-word terms extracted from the corpus; edges between entities that
co-occur inside the same chunk, weighted by co-occurrence count; and chunks
linked to the entities they mention. At query time, the graph retriever
finds question entities, expands one hop to related entities, collects the
chunks attached to those entities, and merges them (RRF) with the vector
ranking. No LLM needed for indexing, which keeps the experiment cheap and
deterministic.

## When GraphRAG actually helps

- Questions about relationships — "which techniques are paired together in
  the docs?" — where two chunks share entities but not vocabulary.
- Multi-hop questions, where the second hop is an entity link, not a lexical
  match.
- Corpus-level questions that no single chunk answers.

## When it does not help

On ordinary single-hop questions the graph adds latency and rarely changes
the answer — the vector ranking already works. GraphRAG is a targeted tool
for relational and global questions, not a default upgrade. This is
quantified in the Phase 6 experiments.

## What the graph actually changes

Under the hood, graph-guided retrieval changes *which chunks get a second
chance*. The vector ranking is still computed; the graph contribution is an
additional ranked list — chunks attached to the question's entities and
their one-hop neighbors — fused in by RRF. Chunks that the vector ranking
buried at rank 30 can surface into the top-5 purely because they share an
entity with the question. That is the mechanism by which the graph catches
hop two: the second document may share no vocabulary with the question at
all, but it shares an entity, and the entity link is enough.

The flip side is equally mechanical: when the question's entities are
common terms, their neighbor chunks flood the fused list and crowd out
precise vector matches — the graph hurts. This explains the experimental
shape: graph retrieval helps on relational and multi-hop questions and
loses ground on ordinary single-hop ones.

## Entity extraction without an LLM

This project's graph is built with a glossary plus capitalized-phrase
frequency: terms in the curated glossary, plus capitalized two- and
three-word phrases appearing in at least two chunks. Edges are
co-occurrence counts inside chunks. This is deliberately weaker than
LLM-based extraction (which finds verbs, attributes, and long-tail
entities) but costs nothing and is deterministic — the right trade for a
learning project whose question is "when does the graph structure itself
help?", independent of extraction quality. A production graph would add
LLM extraction and relation typing on top of the same retrieval loop.
