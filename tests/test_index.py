import pytest

from kb_agent.embedder import HashingEmbedder
from kb_agent.index import BM25Index, KnowledgeIndex, reciprocal_rank_fusion, SearchHit
from kb_agent.schemas import Chunk


def test_bm25_ranks_matching_document_first():
    index = BM25Index().fit(
        ["a", "b", "c"],
        ["混合检索使用 BM25 与向量两路召回", "水果与蔬菜的保存方法", "汽车保养周期"],
    )
    hits = index.search("混合检索", top_k=3)
    assert hits[0].chunk_id == "a"
    assert hits[0].retriever == "bm25"


def test_bm25_returns_empty_for_unknown_term():
    index = BM25Index().fit(["a"], ["水果与蔬菜"])
    assert index.search("量子力学", top_k=3) == []


def test_rrf_merges_both_branches():
    sparse = [SearchHit("x", 1.0, "bm25"), SearchHit("y", 0.5, "bm25")]
    dense = [SearchHit("y", 1.0, "vector")]
    fused = {hit.chunk_id for hit in reciprocal_rank_fusion([sparse, dense])}
    assert fused == {"x", "y"}


def test_rrf_weight_shifts_relative_score():
    sparse = [SearchHit("x", 1.0, "bm25"), SearchHit("y", 0.5, "bm25")]
    dense = [SearchHit("y", 1.0, "vector")]
    balanced = {hit.chunk_id: hit.score for hit in reciprocal_rank_fusion([sparse, dense], weights=[1.0, 1.0])}
    dense_heavy = {hit.chunk_id: hit.score for hit in reciprocal_rank_fusion([sparse, dense], weights=[1.0, 100.0])}
    assert dense_heavy["y"] / dense_heavy["x"] > balanced["y"] / balanced["x"]


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[SearchHit("x", 1.0, "bm25")]], weights=[1.0, 2.0])


def make_chunks() -> list[Chunk]:
    return [
        Chunk("d1#c0", "d1.md", "混合检索", "混合检索用 BM25 与向量两路召回，再用 RRF 融合排名。"),
        Chunk("d2#c0", "d2.md", "切分策略", "固定长度切分按字符数硬切，保留重叠窗口以免句子被截断。"),
    ]


def test_index_round_trips_through_disk(tmp_path):
    embedder = HashingEmbedder(dim=256)
    original = KnowledgeIndex.build(make_chunks(), embedder)
    original.save(tmp_path)

    restored = KnowledgeIndex.load(tmp_path, HashingEmbedder(dim=256))
    assert [chunk.chunk_id for chunk in restored.chunks] == [chunk.chunk_id for chunk in original.chunks]
    assert [hit.chunk_id for hit in restored.search("混合检索", "bm25", 2)] == [
        hit.chunk_id for hit in original.search("混合检索", "bm25", 2)
    ]


def test_index_load_rejects_a_different_embedder(tmp_path):
    """换向量模型后忘了重建索引，必须报错而不是静默给出错结果。"""
    KnowledgeIndex.build(make_chunks(), HashingEmbedder(dim=256)).save(tmp_path)

    with pytest.raises(ValueError, match="重建索引"):
        KnowledgeIndex.load(tmp_path, HashingEmbedder(dim=512))


class _AutoDimEmbedder:
    """模拟远端向量模型：构造时维度未知（auto），要等首次 encode 才确定。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.dim: int | None = None

    def encode(self, texts):
        import numpy as np

        return np.zeros((len(texts), self.dim or 0), dtype=np.float32)


def test_index_load_adopts_dim_when_embedder_dim_is_auto(tmp_path):
    """回归：auto 维度加载索引时以向量文件为准自适应。

    否则按 README 切到真实语义向量后，每次启动都会误判维度不一致，
    触发一次全量重建（重复调用 embedding API，产生费用与启动延迟）。
    """
    KnowledgeIndex.build(make_chunks(), HashingEmbedder(dim=256)).save(tmp_path)

    embedder = _AutoDimEmbedder(name="hashing-256")
    restored = KnowledgeIndex.load(tmp_path, embedder)

    assert embedder.dim == 256
    assert [hit.chunk_id for hit in restored.search("混合检索", "bm25", 2)] == [
        hit.chunk_id for hit in KnowledgeIndex.load(tmp_path, HashingEmbedder(dim=256)).search("混合检索", "bm25", 2)
    ]


def test_index_load_still_rejects_a_wrong_explicit_dim(tmp_path):
    """显式配置了维度且与索引不符时，仍然必须报错。"""
    KnowledgeIndex.build(make_chunks(), HashingEmbedder(dim=256)).save(tmp_path)

    embedder = _AutoDimEmbedder(name="hashing-256")
    embedder.dim = 128
    with pytest.raises(ValueError, match="重建索引"):
        KnowledgeIndex.load(tmp_path, embedder)
