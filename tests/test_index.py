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
