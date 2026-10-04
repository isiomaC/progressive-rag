"""Corpus loading: markdown documents with section structure."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Section:
    level: int
    heading: str
    text: str


@dataclass
class Document:
    id: str            # filename stem, e.g. "02-chunking-strategies"
    title: str
    path: Path
    text: str          # full raw text (headings stripped, paragraphs joined)
    sections: list[Section] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def parse_markdown(path: Path) -> Document:
    """Parse a markdown file into a Document with section structure."""
    raw = path.read_text(encoding="utf-8")
    sections: list[Section] = []
    current_level, current_heading, buf = 1, "", []
    title = path.stem

    for line in raw.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            if buf or current_heading:
                sections.append(Section(current_level, current_heading, "\n".join(buf).strip()))
            current_level, current_heading = len(m.group(1)), m.group(2).strip()
            if not title or title == path.stem:
                title = current_heading
            buf = []
        else:
            buf.append(line)

    if buf or current_heading:
        sections.append(Section(current_level, current_heading, "\n".join(buf).strip()))

    body_parts = [s.text for s in sections if s.text.strip()]
    return Document(
        id=path.stem,
        title=title,
        path=path,
        text="\n\n".join(body_parts),
        sections=sections,
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load all markdown files in sorted filename order (stable ids)."""
    docs = [parse_markdown(p) for p in sorted(corpus_dir.glob("*.md"))]
    if not docs:
        raise FileNotFoundError(f"no markdown corpus files in {corpus_dir}")
    return docs


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter (no NLP deps)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'`(])", text.strip())
    return [p.strip() for p in parts if p.strip()]
