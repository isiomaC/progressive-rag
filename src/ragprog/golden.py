"""Golden dataset loading + per-strategy relevant-chunk resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .chunking import Chunk
from .eval_metrics import bigram_f1


@dataclass
class GoldenItem:
    id: str
    type: str                       # single-hop | multi-hop | tricky | comparative
    question: str
    answer: str
    docs: list[str]                 # corpus doc stems containing the answer

    @property
    def relevant_doc_ids(self) -> set[str]:
        return set(self.docs)


def load_golden(path: Path) -> list[GoldenItem]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        GoldenItem(
            id=it["id"], type=it["type"], question=it["question"],
            answer=it["answer"], docs=list(it.get("docs") or []),
        )
        for it in data["items"]
    ]


def resolve_relevant_chunks(golden: list[GoldenItem], chunks: list[Chunk]) -> dict[str, set[str]]:
    """Map each golden question to concrete chunk ids for THIS chunking
    strategy: per required doc, the chunk with the highest lexical overlap
    with the ground-truth answer."""
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    resolved: dict[str, set[str]] = {}
    for item in golden:
        ids: set[str] = set()
        for doc_id in item.docs:
            candidates = by_doc.get(doc_id, [])
            if not candidates:
                continue
            best = max(candidates, key=lambda c: bigram_f1(item.answer, c.text))
            if bigram_f1(item.answer, best.text) > 0.05:
                ids.add(best.id)
        resolved[item.id] = ids
    return resolved
