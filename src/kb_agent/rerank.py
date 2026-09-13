"""重排。

默认实现是启发式的：把 idf 加权的查询词覆盖率、原文短语命中、标题命中与融合分数
线性组合。它不是 cross-encoder，但在小规模中文知识库上能稳定提升头部命中，
且完全离线、可解释、毫秒级。需要更强效果时换成 CrossEncoderReranker。
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

from .schemas import RetrievedChunk
from .text import normalize, tokenize

_WHITESPACE_RE = re.compile(r"\s+")


def longest_common_substring_ratio(query: str, text: str) -> float:
    """查询与文本的最长公共子串占查询长度的比例。

    原来用"整句查询是否原样出现在原文里"作为短语特征，中文长问句几乎永远不命中，
    这个特征恒为 0，白白占掉权重。最长公共子串对部分命中同样敏感：
    "切分粒度" 命中原文的"切分粒度太粗"就能拿到分。
    """
    a = _WHITESPACE_RE.sub("", normalize(query))
    b = _WHITESPACE_RE.sub("", normalize(text))
    if not a or not b:
        return 0.0
    positions: dict[str, list[int]] = {}
    for index, char in enumerate(b, start=1):
        positions.setdefault(char, []).append(index)
    previous = [0] * (len(b) + 1)
    best = 0
    for char in a:
        current = [0] * (len(b) + 1)
        for index in positions.get(char, ()):  # 只遍历可能匹配的位置，避免 O(n*m) 全量扫描
            value = previous[index - 1] + 1
            current[index] = value
            if value > best:
                best = value
        previous = current
    return best / len(a)


def idf_weighted_coverage(query_terms: set[str], text_terms: set[str], idf: dict[str, float]) -> float:
    """用逆文档频率加权的查询词覆盖率。

    直接用词覆盖率会让"怎么""什么"这类高频词主导判断，idf 加权后
    真正稀有的技术词（专有名词、缩写）才是决定性的，这一点同时被重排和拒答门控复用。
    """
    denominator = sum(idf.get(term, 1.0) for term in query_terms)
    if denominator <= 0:
        return 0.0
    numerator = sum(idf.get(term, 1.0) for term in query_terms if term in text_terms)
    return numerator / denominator


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...


class NoopReranker:
    """对照组：不做重排，直接截断。"""

    name = "noop"

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        for rank, candidate in enumerate(candidates, start=1):
            candidate.rank = rank
        return list(candidates[:top_k])


class HeuristicReranker:
    name = "heuristic"

    def __init__(
        self,
        idf: dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.idf = idf or {}
        self.weights = weights or {"base": 0.30, "coverage": 0.35, "overlap": 0.25, "title": 0.10}

    def _weighted_coverage(self, query_terms: set[str], text_terms: set[str]) -> float:
        return idf_weighted_coverage(query_terms, text_terms, self.idf)

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        query_terms = set(tokenize(query))
        best_base = max(candidate.score for candidate in candidates) or 1.0
        rescored: list[RetrievedChunk] = []
        for candidate in candidates:
            text = candidate.chunk.text
            text_terms = set(tokenize(text))
            features = {
                "base": candidate.score / best_base,
                "coverage": self._weighted_coverage(query_terms, text_terms),
                "overlap": longest_common_substring_ratio(query, text),
                "title": self._weighted_coverage(query_terms, set(tokenize(candidate.chunk.title))),
            }
            score = sum(self.weights[name] * value for name, value in features.items())
            candidate.score = score
            rescored.append(candidate)
        rescored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        for rank, candidate in enumerate(rescored, start=1):
            candidate.rank = rank
            candidate.retriever = f"{candidate.retriever}+{self.name}"
        return rescored[:top_k]


class CrossEncoderReranker:
    """可选：装了 sentence-transformers 才可用，用于替换启发式版本。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self.name = f"cross-encoder:{model_name}"

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = self._model.predict(pairs)
        rescored = list(candidates)
        for candidate, score in zip(rescored, scores):
            candidate.score = float(score)
        rescored.sort(key=lambda item: -item.score)
        for rank, candidate in enumerate(rescored, start=1):
            candidate.rank = rank
        return rescored[:top_k]
