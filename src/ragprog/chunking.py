"""Chunking strategies — Phase 2 of the project.

Four strategies are compared on identical corpora and evaluation sets:
  FixedChunker        — fixed-size words + overlap (the Phase 1 baseline)
  RecursiveChunker    — separator hierarchy (paragraphs → sentences → words)
  SemanticChunker     — sentence-buffer similarity breaks
  ParentChildChunker  — small children for retrieval, big parents for prompts

Chunk sizes are measured in words (a cheap, deterministic token proxy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .corpus import Document, split_sentences

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    start: int          # character offset in the document text
    end: int
    parent_id: str | None = None

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class Chunker:
    """Base class. Subclasses implement chunk(doc) -> list[Chunk]."""

    name = "base"

    def chunk(self, doc: Document) -> list[Chunk]:
        raise NotImplementedError

    def _make(self, doc: Document, pieces: list[tuple[int, int, str]],
              parent: str | None = None) -> list[Chunk]:
        return [
            Chunk(id=f"{doc.id}::{i}", doc_id=doc.id, text=text,
                  start=s, end=e, parent_id=parent)
            for i, (s, e, text) in enumerate(pieces)
            if text.strip()
        ]


def _fixed_pieces(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    pieces, step = [], max(1, size - overlap)
    i = 0
    while i < len(words):
        window = words[i:i + size]
        if not window:
            break
        pieces.append(" ".join(window))
        if i + size >= len(words):
            break
        i += step
    return pieces


class FixedChunker(Chunker):
    name = "fixed"

    def __init__(self, size: int = 500, overlap: int = 50):
        self.size, self.overlap = size, overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces = _fixed_pieces(doc.text, self.size, self.overlap)
        spans = _locate(doc.text, pieces)
        return self._make(doc, spans)


class RecursiveChunker(Chunker):
    """Split on a hierarchy of separators, smallest fallback last.

    Paragraphs stay intact wherever possible; only oversized paragraphs are
    broken down to sentences and words. This preserves the author's
    structure — the key advantage over fixed chunking.
    """

    name = "recursive"

    def __init__(self, size: int = 500, overlap: int = 50,
                 separators: list[str] | None = None):
        self.size, self.overlap = size, overlap
        self.separators = separators or SEPARATORS

    def _split(self, text: str, depth: int) -> list[str]:
        if len(text.split()) <= self.size:
            return [text]
        if depth >= len(self.separators):
            # no separator can split further: fall back to fixed windows
            return _fixed_pieces(text, self.size, self.overlap)
        sep = self.separators[depth]
        parts = text.split(sep)
        if len(parts) == 1:
            return self._split(text, depth + 1)
        out: list[str] = []
        buf = ""
        for part in parts:
            candidate = f"{buf}{sep}{part}" if buf else part
            if len(candidate.split()) > self.size and buf:
                out.extend(self._split(buf, depth + 1))
                buf = part
            else:
                buf = candidate
        if buf:
            out.extend(self._split(buf, depth + 1))
        return [p for p in out if p.strip()]

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces = self._split(doc.text, 0)
        return self._make(doc, _locate(doc.text, pieces))


class SemanticChunker(Chunker):
    """Group sentences by embedding similarity (buffer vs next sentence).

    Break when similarity drops below the percentile threshold of all
    candidate breakpoints in the document — variable-length, topically
    coherent chunks. Costs extra embedding calls at index time only.
    """

    name = "semantic"

    def __init__(self, embedder, percentile: int = 90):
        self.embedder = embedder
        self.percentile = percentile

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = split_sentences(doc.text)
        if len(sentences) <= 1:
            return self._make(doc, _locate(doc.text, [doc.text]))
        embs = self.embedder.embed(sentences)

        # similarity between sentence i and i+1 → candidate breakpoints
        sims = np.diag(embs[:-1] @ embs[1:].T)
        threshold = float(np.percentile(sims, self.percentile)) if len(sims) else 0.0

        chunks, buf = [], [sentences[0]]
        for i in range(1, len(sentences)):
            if sims[i - 1] < threshold and buf:
                chunks.append(" ".join(buf))
                buf = [sentences[i]]
            else:
                buf.append(sentences[i])
        if buf:
            chunks.append(" ".join(buf))
        return self._make(doc, _locate(doc.text, chunks))


class ParentChildChunker(Chunker):
    """Small-to-big: index children (~250w), prompt with parents (~1024w).

    Each child maps to the parent that contains it; several children can
    share one parent. The retrieval-side mapping lives in retrieval.py.
    """

    name = "parent-child"

    def __init__(self, child_size: int = 250, parent_size: int = 1024):
        self.child_size, self.parent_size = child_size, parent_size

    def chunk(self, doc: Document) -> list[Chunk]:
        parents = RecursiveChunker(self.parent_size, 0).chunk(doc)
        children = RecursiveChunker(self.child_size, 0).chunk(doc)
        # map each child to the parent containing its midpoint
        mapped = []
        for c in children:
            mid = (c.start + c.end) // 2
            pid = next((p.id for p in parents if p.start <= mid < p.end), parents[-1].id if parents else None)
            mapped.append(Chunk(id=c.id, doc_id=c.doc_id, text=c.text,
                                start=c.start, end=c.end, parent_id=pid))
        return parents + mapped


def _locate(text: str, pieces: list[str]) -> list[tuple[int, int, str]]:
    """Recover character offsets for pieces by scanning the text."""
    spans, pos = [], 0
    for piece in pieces:
        idx = text.find(piece, pos)
        if idx == -1:
            idx = text.find(piece)
        if idx == -1:
            spans.append((pos, pos + len(piece), piece))
            continue
        spans.append((idx, idx + len(piece), piece))
        pos = idx + len(piece)
    return spans
