import pytest

from kb_agent.rerank import HeuristicReranker, longest_common_substring_ratio
from kb_agent.schemas import Chunk, RetrievedChunk


def make_candidate(chunk_id: str, text: str, score: float, rank: int) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(chunk_id, "d.md", "标题", text), score=score, rank=rank)


def test_longest_common_substring_ratio():
    assert longest_common_substring_ratio("文档切分粒度", "关于文档切分的说明") == pytest.approx(4 / 6)


def test_longest_common_substring_handles_no_overlap():
    assert longest_common_substring_ratio("混合检索", "汽车保养") == 0.0


def test_reranker_promotes_relevant_candidate_over_higher_scored_noise():
    query = "切分粒度太细会有什么问题"
    relevant = make_candidate("relevant", "切分粒度太细会破坏语义完整性", score=0.5, rank=2)
    noise = make_candidate("noise", "这是一段完全无关的文字内容", score=1.0, rank=1)
    top = HeuristicReranker(idf={}).rerank(query, [noise, relevant], top_k=1)
    assert top[0].chunk.chunk_id == "relevant"


def test_reranker_records_rank_and_retriever():
    query = "混合检索"
    candidates = [make_candidate("a", "混合检索与重排", 0.9, 1), make_candidate("b", "无关内容", 0.1, 2)]
    result = HeuristicReranker(idf={}).rerank(query, candidates, top_k=2)
    assert [item.rank for item in result] == [1, 2]
    assert all("heuristic" in item.retriever for item in result)
