"""检索管线：召回 → 融合 → 重排 → 组装上下文。"""

from __future__ import annotations

from dataclasses import dataclass

from .index import KnowledgeIndex
from .rerank import NoopReranker, Reranker
from .schemas import RetrievedChunk, TraceSpan


@dataclass
class RetrievalResult:
    query: str
    rewritten_query: str
    hits: list[RetrievedChunk]
    spans: list[TraceSpan]


class Retriever:
    def __init__(
        self,
        index: KnowledgeIndex,
        reranker: Reranker | None = None,
        mode: str = "hybrid",
        candidate_k: int = 20,
        top_k: int = 5,
    ) -> None:
        self.index = index
        self.reranker = reranker or NoopReranker()
        self.mode = mode
        self.candidate_k = candidate_k
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        use_rerank: bool = True,
    ) -> list[RetrievedChunk]:
        mode = mode or self.mode
        top_k = top_k or self.top_k
        hits = self.index.search(query, mode=mode, candidate_k=self.candidate_k)
        candidates: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits, start=1):
            chunk = self.index.chunk_map.get(hit.chunk_id)
            if chunk is None:
                continue
            parent_text = None
            if chunk.parent_id:
                parent = self.index.chunk_map.get(chunk.parent_id)
                parent_text = parent.text if parent else None
            candidates.append(
                RetrievedChunk(chunk=chunk, score=hit.score, retriever=hit.retriever, rank=rank, parent_text=parent_text)
            )
        reranker = self.reranker if use_rerank else NoopReranker()
        return reranker.rerank(query, candidates, top_k)
