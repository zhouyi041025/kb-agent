"""领域模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str = ""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int = 0
    parent_id: str | None = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    retriever: str = ""
    rank: int = 0
    parent_text: str | None = None

    @property
    def citation_text(self) -> str:
        return self.parent_text or self.chunk.text


@dataclass
class TraceSpan:
    name: str
    duration_ms: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    contexts: list[RetrievedChunk] = field(default_factory=list)
    degraded: bool = False
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    spans: list[TraceSpan] = field(default_factory=list)

    @property
    def latency_ms(self) -> float:
        return round(sum(span.duration_ms for span in self.spans), 2)
