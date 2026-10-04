"""GraphRAG — Phase 6.2 (simplified LlamaIndex-KG style, no LLM indexing).

Builds an entity graph from the corpus:
  - entities: curated glossary + capitalized multi-word terms (freq ≥ 2)
  - entity→chunks edges (mention)
  - entity↔entity edges (co-occurrence in the same chunk, weighted)

At query time: question entities → one-hop neighbors → their chunks,
RRF-merged with the vector ranking. This is the lightweight family of
GraphRAG; see docs/LEARNING.md for the contrast with Microsoft's
community-summary GraphRAG.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from .config import GRAPH_GLOSSARY
from .retrieval import ScoredChunk, rrf_merge
from .stores import CorpusIndex

CAPS_RE = re.compile(r"\b([A-Z][a-z0-9-]+(?:\s+[A-Z][a-z0-9-]+){0,2})\b")

_STOP_CAPS = {
    "The", "This", "That", "These", "Those", "Context", "Question", "Answer",
    "However", "Therefore", "Because", "If", "In", "On", "It", "As", "A",
    "When", "While", "What", "Which", "Every", "Each", "Any", "No", "Not",
    "You", "We", "They", "He", "She", "Phase", "Chunk", "All", "With",
    "Note", "Intro", "None", "I", "For", "But", "And", "Or", "From",
    "See", "Since", "Its", "Their", "Our", "Using", "Then", "There",
}


class KnowledgeGraph:
    def __init__(self):
        self.entity_chunks: dict[str, set[str]] = defaultdict(set)
        self.co_occurrence: Counter = Counter()
        self.entities: list[str] = []

    @classmethod
    def build(cls, chunks) -> "KnowledgeGraph":
        kg = cls()
        texts = {c.id: c.text for c in chunks}

        # candidate entities: glossary + frequent capitalized phrases
        cap_freq: Counter = Counter()
        for text in texts.values():
            for m in CAPS_RE.finditer(text):
                phrase = m.group(1).strip()
                if phrase not in _STOP_CAPS and len(phrase.split()) <= 3:
                    cap_freq[phrase] += 1
        candidates = set(GRAPH_GLOSSARY)
        candidates |= {p for p, f in cap_freq.items() if f >= 2}
        # drop candidates that only match glossary case-sensitively
        candidates = {c for c in candidates if any(c.lower() in t.lower() for t in texts.values())}

        for cid, text in texts.items():
            low = text.lower()
            for ent in candidates:
                if ent.lower() in low:
                    kg.entity_chunks[ent].add(cid)

        entities = sorted(candidates, key=lambda e: len(kg.entity_chunks[e]), reverse=True)
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                shared = kg.entity_chunks[e1] & kg.entity_chunks[e2]
                if shared:
                    kg.co_occurrence[(e1, e2)] += len(shared)

        kg.entities = entities
        return kg

    def entities_in(self, text: str) -> list[str]:
        low = text.lower()
        return [e for e in self.entities if e.lower() in low]

    def neighbors(self, entity: str, max_n: int = 5) -> list[tuple[str, int]]:
        out = []
        for (a, b), w in self.co_occurrence.items():
            if a == entity:
                out.append((b, w))
            elif b == entity:
                out.append((a, w))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:max_n]


class GraphRetriever:
    name = "graph"

    def __init__(self, index: CorpusIndex, graph: KnowledgeGraph):
        self.index, self.graph = index, graph

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        graph_scores: dict[str, float] = {}
        seen_entities: set[str] = set()
        for ent in self.graph.entities_in(question):
            if ent in seen_entities:
                continue
            seen_entities.add(ent)
            for cid in self.graph.entity_chunks[ent]:
                graph_scores[cid] = graph_scores.get(cid, 0.0) + 1.0
            for nb, w in self.graph.neighbors(ent):
                if nb in seen_entities:
                    continue
                for cid in self.graph.entity_chunks[nb]:
                    graph_scores[cid] = graph_scores.get(cid, 0.0) + 0.5 + 0.05 * w

        vec = self.index.search_vector(question, max(k * 4, 20))
        graph_ranked = sorted(graph_scores.items(), key=lambda x: x[1], reverse=True)[: max(k * 4, 20)]
        fused = rrf_merge([vec, graph_ranked])[:k]

        out = []
        for cid, s in fused:
            c = self.index.get(cid)
            out.append(ScoredChunk(c.id, c.doc_id, c.text, s, "graph", c.parent_id))
        return out
