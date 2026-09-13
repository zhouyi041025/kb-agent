"""检索索引：BM25（稀疏）+ 向量（稠密）+ RRF 融合，含磁盘持久化。

BM25 与向量索引都自己实现，不依赖外部检索服务：这样索引怎么算、分数怎么融合
都是可见的，评测结果也能逐条解释。生产环境把 VectorIndex 换成 Milvus/FAISS
只需要保持 search() 的签名。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .embedder import Embedder
from .schemas import Chunk
from .text import tokenize


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    retriever: str


class BM25Index:
    """Okapi BM25，带倒排索引，中文按 bigram 检索。"""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = []
        self.doc_terms: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl = 1.0
        self.idf: dict[str, float] = {}
        self.inverted: dict[str, list[tuple[int, int]]] = {}

    def fit(self, chunk_ids: Sequence[str], texts: Sequence[str]) -> "BM25Index":
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids 与 texts 长度不一致")
        self.chunk_ids = list(chunk_ids)
        self.doc_terms = [tokenize(text) for text in texts]
        self.doc_len = [len(terms) for terms in self.doc_terms]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0

        document_frequency: dict[str, int] = {}
        for index, terms in enumerate(self.doc_terms):
            counts: dict[str, int] = {}
            for term in terms:
                counts[term] = counts.get(term, 0) + 1
            for term, frequency in counts.items():
                document_frequency[term] = document_frequency.get(term, 0) + 1
                self.inverted.setdefault(term, []).append((index, frequency))

        total = len(self.doc_terms)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        return self

    def search(self, query: str, top_k: int = 20) -> list[SearchHit]:
        if not self.doc_terms:
            return []
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_index, frequency in postings:
                length_ratio = self.doc_len[doc_index] / self.avgdl if self.avgdl else 1.0
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[doc_index] = scores.get(doc_index, 0.0) + idf * frequency * (self.k1 + 1) / denominator
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [SearchHit(self.chunk_ids[index], float(score), "bm25") for index, score in ranked]


class VectorIndex:
    """余弦相似度的扁平索引。向量已 L2 归一化，内积即余弦。"""

    def __init__(self) -> None:
        self.chunk_ids: list[str] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def fit(self, chunk_ids: Sequence[str], vectors: np.ndarray) -> "VectorIndex":
        if vectors.ndim != 2:
            raise ValueError("vectors 必须是二维数组")
        if len(chunk_ids) != vectors.shape[0]:
            raise ValueError("chunk_ids 与 vectors 行数不一致")
        self.chunk_ids = list(chunk_ids)
        self.matrix = vectors.astype(np.float32, copy=False)
        return self

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[SearchHit]:
        if not self.chunk_ids or self.matrix.size == 0:
            return []
        scores = self.matrix @ query_vector.astype(np.float32, copy=False)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [SearchHit(self.chunk_ids[index], float(scores[index]), "vector") for index in order]


def reciprocal_rank_fusion(
    result_lists: Iterable[Sequence[SearchHit]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[SearchHit]:
    """RRF：只用排名融合多路召回，避免不同检索器的分数量纲不可比。

    weights 支持按分支质量加权。等权在"两路水平相当"时最稳；当一路明显更强时，
    等权会把强分支的头部结果冲淡，这时需要调低弱分支的权重（见 eval 的权重消融）。
    """
    lists = list(result_lists)
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError("weights 数量必须与召回分支数量一致")
    fused: dict[str, float] = {}
    for hits, weight in zip(lists, weights):
        for rank, hit in enumerate(hits, start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + weight / (k + rank)
    ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [SearchHit(chunk_id, float(score), "hybrid") for chunk_id, score in ranked]


class KnowledgeIndex:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder, fusion_weights: Sequence[float] | None = None) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        # [稀疏 BM25 权重, 稠密向量权重]
        self.fusion_weights = list(fusion_weights) if fusion_weights else [1.0, 1.0]
        self.chunk_map = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.bm25 = BM25Index()
        self.vectors = VectorIndex()

    @classmethod
    def build(cls, chunks: Sequence[Chunk], embedder: Embedder, batch_size: int = 64) -> "KnowledgeIndex":
        index = cls(chunks, embedder)
        chunk_ids = [chunk.chunk_id for chunk in index.chunks]
        index.bm25.fit(chunk_ids, [chunk.text for chunk in index.chunks])
        vectors = []
        texts = [chunk.text for chunk in index.chunks]
        for start in range(0, len(texts), batch_size):
            vectors.append(embedder.encode(texts[start : start + batch_size]))
        matrix = np.vstack(vectors) if vectors else np.zeros((0, embedder.dim), dtype=np.float32)
        index.vectors.fit(chunk_ids, matrix)
        return index

    def search(self, query: str, mode: str = "hybrid", candidate_k: int = 20) -> list[SearchHit]:
        if mode == "bm25":
            return self.bm25.search(query, candidate_k)
        if mode == "vector":
            query_vector = self.embedder.encode([query])[0]
            return self.vectors.search(query_vector, candidate_k)
        if mode == "hybrid":
            sparse = self.bm25.search(query, candidate_k)
            dense = self.vectors.search(self.embedder.encode([query])[0], candidate_k)
            return reciprocal_rank_fusion([sparse, dense], weights=self.fusion_weights)[:candidate_k]
        raise ValueError(f"未知检索模式：{mode}")

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        np.save(directory / "vectors.npy", self.vectors.matrix)
        meta = {
            "count": len(self.chunks),
            "embedder": self.embedder.name,
            "dim": int(self.vectors.matrix.shape[1]) if self.vectors.matrix.size else self.embedder.dim,
        }
        (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> "KnowledgeIndex":
        directory = Path(directory)
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"索引不存在：{directory}，请先运行 scripts/build_index.py")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks = [
            Chunk(**json.loads(line))
            for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        index = cls(chunks, embedder)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        index.bm25.fit(chunk_ids, [chunk.text for chunk in chunks])
        if meta.get("embedder") and meta["embedder"] != embedder.name:
            raise ValueError(
                f"索引是用 {meta['embedder']} 建的，当前 embedder 是 {embedder.name}，请重建索引"
            )
        matrix = np.load(directory / "vectors.npy")
        if matrix.shape[1] != embedder.dim:
            raise ValueError(
                f"索引向量维度 {matrix.shape[1]} 与当前 embedder 维度 {embedder.dim} 不一致，请重建索引"
            )
        index.vectors.fit(chunk_ids, matrix)
        return index
