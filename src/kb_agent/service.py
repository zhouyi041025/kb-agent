"""编排层：把改写、检索、生成、缓存、降级、追踪串成一次问答。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .cache import QueryCache
from .config import AppConfig
from .embedder import build_embedder
from .index import KnowledgeIndex
from .ingest import build_chunks, load_documents
from .llm import LLMClient, StubLLM, build_llm
from .prompt import Message, build_messages
from .rerank import HeuristicReranker, Reranker, idf_weighted_coverage
from .retriever import Retriever
from .rewrite import HeuristicRewriter, QueryRewriter
from .schemas import Answer, RetrievedChunk
from .tracing import Tracer
from .text import tokenize

_CITATION_RE = re.compile(r"\[(\d+)\]")
_CITATION_LABEL = r"(?:编号|引用|来源|文档|资料|参考)"
_CITATION_OPEN = r"(?:\[|【|［)"
_CITATION_CLOSE = r"(?:\]|】|］)"
_CITATION_GROUP_RE = re.compile(
    rf"{_CITATION_OPEN}\s*{_CITATION_LABEL}?\s*[:：]?\s*(\d+(?:\s*[,、/]\s*\d+)+)\s*{_CITATION_CLOSE}"
)
_CITATION_SINGLE_RE = re.compile(
    rf"{_CITATION_OPEN}\s*{_CITATION_LABEL}?\s*[:：]?\s*(\d+)\s*{_CITATION_CLOSE}"
)
REFUSAL_TEXT = "知识库中没有找到相关依据，无法回答该问题。"


def normalize_citations(text: str) -> str:
    """把模型五花八门的引用写法统一成 `[数字]`。

    小模型经常把提示词里的占位符照抄下来，写成「[编号1]」「【编号：1】」甚至
    「[1、2]」。不归一化的话，引用解析会静默落空：答案看着没问题，但前端
    高亮不出原文、引用准确率也统计不到 —— 这类"格式没对齐"比答案错更难发现。
    """
    text = _CITATION_GROUP_RE.sub(
        lambda match: "".join(f"[{n}]" for n in re.findall(r"\d+", match.group(1))), text
    )
    return _CITATION_SINGLE_RE.sub(r"[\1]", text)


@dataclass
class ServiceStats:
    requests: int = 0
    degraded: int = 0
    refusals: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms_total: float = 0.0
    cache: dict = field(default_factory=dict)

    def as_dict(self, cache: QueryCache | None = None) -> dict:
        average = self.latency_ms_total / self.requests if self.requests else 0.0
        payload = {
            "requests": self.requests,
            "degraded": self.degraded,
            "refusals": self.refusals,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "avg_latency_ms": round(average, 2),
        }
        if cache is not None:
            payload["cache"] = {"size": cache.size, "hits": cache.hits, "misses": cache.misses, "hit_rate": cache.hit_rate}
        return payload


class KbService:
    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        index: KnowledgeIndex | None = None,
        llm: LLMClient | None = None,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
        cache: QueryCache | None = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self.embedder = build_embedder(
            self.config.embedding,
            base_url=self.config.llm.base_url,
            api_key=self.config.llm.api_key,
        )
        self.index = index
        self.llm = llm or build_llm(self.config.llm)
        self.reranker = reranker or HeuristicReranker()
        self.rewriter = rewriter or HeuristicRewriter()
        self.cache = cache or QueryCache(max_size=self.config.cache_size)
        self.stats = ServiceStats()
        self._retriever: Retriever | None = None

    # ---------- 索引 ----------

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            if self.index is None:
                raise RuntimeError("索引尚未加载，请先调用 load_index() 或 ingest()")
            idf = self.index.bm25.idf
            if isinstance(self.reranker, HeuristicReranker) and not self.reranker.idf:
                self.reranker.idf = idf
            self._retriever = Retriever(
                self.index,
                reranker=self.reranker,
                mode=self.config.retrieval.mode,
                candidate_k=self.config.retrieval.candidate_k,
                top_k=self.config.retrieval.top_k,
            )
        return self._retriever

    def ingest(self, directory: str | Path | None = None, *, save: bool = True) -> dict:
        directory = Path(directory or self.config.knowledge_dir)
        documents = load_documents(directory)
        chunks = build_chunks(documents, strategy="recursive")
        self.index = KnowledgeIndex.build(chunks, self.embedder)
        self._retriever = None
        if save:
            self.index.save(self.config.index_dir)
        return {"documents": len(documents), "chunks": len(chunks), "index_dir": str(self.config.index_dir) if save else None}

    def load_index(self) -> None:
        self.index = KnowledgeIndex.load(self.config.index_dir, self.embedder)
        self._retriever = None

    def ensure_index(self, rebuild: bool = False) -> None:
        if self.index is not None and not rebuild:
            return
        if not rebuild:
            try:
                self.load_index()
                return
            except (FileNotFoundError, ValueError):
                pass
        self.ingest()

    # ---------- 问答 ----------

    def answer(
        self,
        question: str,
        history: Sequence[Message] | None = None,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        use_cache: bool = True,
    ) -> Answer:
        mode = mode or self.config.retrieval.mode
        top_k = top_k or self.config.retrieval.top_k
        use_rerank = self.config.retrieval.use_rerank if use_rerank is None else use_rerank

        cache_key = QueryCache.make_key(question, mode, top_k, use_rerank, history)
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.stats.requests += 1
                cached.cached = True
                self.stats.latency_ms_total += cached.latency_ms
                return cached

        tracer = Tracer()
        with tracer.span("rewrite"):
            rewritten = self.rewriter.rewrite(question, history or [])

        with tracer.span("retrieve", mode=mode):
            contexts = self.retriever.retrieve(rewritten, top_k=top_k, mode=mode, use_rerank=use_rerank)

        confidence = self.confidence(rewritten, contexts)
        tracer.record("confidence", 0.0, value=round(confidence, 4))

        if not contexts or confidence < self.config.retrieval.min_confidence:
            # 拒答也要把召回结果带回去：前端可以展示"找到了这些资料但依据不足"，
            # 同时保证评测里的召回指标不受拒答策略影响（两件事必须能分开度量）。
            answer = Answer(question=question, answer=REFUSAL_TEXT, contexts=contexts, spans=tracer.spans)
            self._finalize(answer, cache_key, use_cache)
            return answer

        messages = build_messages(question, contexts, history)
        degraded = False
        with tracer.span("generate", model=self.llm.name):
            try:
                result = self.llm.complete(messages, contexts=contexts, question=question)
            except Exception:
                # 降级：模型不可用时退回抽取式回答，宁可答得朴素也不返回 5xx
                degraded = True
                result = StubLLM().complete(messages, contexts=contexts, question=question)

        text = normalize_citations(result.text)
        citations = self._resolve_citations(text, contexts)
        answer = Answer(
            question=question,
            answer=text,
            citations=citations,
            contexts=contexts,
            degraded=degraded,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
            spans=tracer.spans,
        )
        self._finalize(answer, cache_key, use_cache)
        return answer

    def _finalize(self, answer: Answer, cache_key: str, use_cache: bool) -> None:
        self.stats.requests += 1
        self.stats.latency_ms_total += answer.latency_ms
        self.stats.prompt_tokens += answer.prompt_tokens
        self.stats.completion_tokens += answer.completion_tokens
        self.stats.cost_usd += answer.cost_usd
        if answer.degraded:
            self.stats.degraded += 1
        if answer.answer.strip() == REFUSAL_TEXT:
            self.stats.refusals += 1
        if use_cache and not answer.cached:
            self.cache.set(cache_key, answer)

    @staticmethod
    def _resolve_citations(text: str, contexts: Sequence[RetrievedChunk]) -> list[str]:
        """把答案里的 [1][2] 映射回 chunk_id，供前端高亮原文和评测引用准确率。"""
        citations: list[str] = []
        for match in _CITATION_RE.findall(normalize_citations(text)):
            position = int(match) - 1
            if 0 <= position < len(contexts):
                chunk_id = contexts[position].chunk.chunk_id
                if chunk_id not in citations:
                    citations.append(chunk_id)
        return citations

    def confidence(self, query: str, contexts: Sequence[RetrievedChunk]) -> float:
        """前若干片段词项并集的 idf 加权覆盖率，作为"知识库里到底有没有答案"的判据。

        用并集而不是单个最佳片段：多跳问题（例如"混合检索和重排先上哪个"）的答案
        分散在多个片段里，单片段判据会把它们误判成无依据，导致漏答率显著偏高。
        """
        if not contexts or self.index is None:
            return 0.0
        query_terms = set(tokenize(query))
        if not query_terms:
            return 0.0
        window = max(1, self.config.retrieval.confidence_context_window)
        union_terms: set[str] = set()
        for context in contexts[:window]:
            union_terms |= set(tokenize(context.chunk.text))
        return idf_weighted_coverage(query_terms, union_terms, self.index.bm25.idf)

    def stats_payload(self) -> dict:
        return self.stats.as_dict(self.cache)
