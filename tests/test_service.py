import pytest

from kb_agent.config import AppConfig
from kb_agent.embedder import HashingEmbedder
from kb_agent.index import KnowledgeIndex
from kb_agent.llm import LLMResult, StubLLM
from kb_agent.schemas import Chunk
from kb_agent.service import REFUSAL_TEXT, KbService, normalize_citations

QUESTION = "混合检索是怎么融合的"


def make_index() -> KnowledgeIndex:
    chunks = [
        Chunk(
            "d1#c0",
            "d1.md",
            "混合检索",
            "混合检索使用 BM25 与向量两路召回，再用 RRF 融合排名，避免分数量纲不一致。",
        ),
        Chunk("d2#c0", "d2.md", "切分策略", "递归切分按标题和段落切分，尽量保持段落完整。"),
    ]
    return KnowledgeIndex.build(chunks, HashingEmbedder(dim=256))


def make_service(llm=None, min_confidence: float = 0.0, **overrides) -> KbService:
    """单测用两篇文档的小语料，idf 统计没有代表性，会让置信度被系统性压低。
    因此默认关闭门控，门控行为由 test_confidence_gate_* 用显式阈值单独验证。
    真实语料上的阈值标定见 eval/report.md。
    """
    config = AppConfig()
    config.retrieval.min_confidence = min_confidence
    for key, value in overrides.items():
        setattr(config.retrieval, key, value)
    return KbService(config, index=make_index(), llm=llm or StubLLM())


def test_answers_when_evidence_exists():
    answer = make_service().answer(QUESTION)
    assert answer.answer != REFUSAL_TEXT
    assert answer.citations
    assert answer.contexts


def test_refuses_when_knowledge_base_has_no_evidence():
    answer = make_service(min_confidence=0.45).answer("公司年假一共有多少天")
    assert answer.answer == REFUSAL_TEXT


def test_refusal_still_returns_retrieved_contexts():
    """拒答不能丢掉召回结果：前端要展示"找到了这些资料"，评测也要能分开度量召回与拒答。"""
    answer = make_service(min_confidence=0.45).answer("公司年假一共有多少天")
    assert answer.contexts


def test_confidence_gate_blocks_generation_below_threshold():
    """用一个"无论如何都回答"的假模型，把门控行为从抽取式模型的拒答里隔离出来测。"""

    class EchoLLM:
        name = "echo"

        def complete(self, messages, contexts, question):
            return LLMResult(text="这是模型的回答 [1]")

    blocked = make_service(llm=EchoLLM(), min_confidence=0.35).answer("公司年假一共有多少天")
    assert blocked.answer == REFUSAL_TEXT

    allowed = make_service(llm=EchoLLM(), min_confidence=0.0).answer("公司年假一共有多少天")
    assert allowed.answer != REFUSAL_TEXT
    assert allowed.degraded is False


def test_degrades_when_llm_fails():
    class BrokenLLM:
        name = "broken"

        def complete(self, *args, **kwargs):
            raise RuntimeError("上游超时")

    answer = make_service(llm=BrokenLLM()).answer(QUESTION)
    assert answer.degraded is True
    assert answer.answer != REFUSAL_TEXT


def test_cache_hit_on_repeated_question():
    service = make_service()
    service.answer(QUESTION)
    second = service.answer(QUESTION)
    assert second.cached is True
    assert service.cache.hits == 1


def test_trace_records_each_stage():
    answer = make_service().answer(QUESTION)
    stage_names = {span.name for span in answer.spans}
    assert {"rewrite", "retrieve", "generate"} <= stage_names
    assert answer.latency_ms > 0


def test_stats_accumulate():
    service = make_service()
    service.answer(QUESTION)
    service.answer("公司年假一共有多少天")
    payload = service.stats_payload()
    assert payload["requests"] == 2
    # 第二个问题在语料里没有任何依据，门控关闭时由抽取式模型自己拒答
    assert payload["refusals"] == 1


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        make_service().answer(QUESTION, mode="不存在")


def test_normalize_citations_accepts_common_variants():
    """小模型会把提示词占位符照抄成 [编号1]，也会写全角括号、多个编号挤一格。"""
    assert normalize_citations("结论 [编号1]") == "结论 [1]"
    assert normalize_citations("结论【编号：2】") == "结论[2]"
    assert normalize_citations("结论 [引用 3] 与 [来源4]") == "结论 [3] 与 [4]"
    assert normalize_citations("结论 [1、2]") == "结论 [1][2]"
    assert normalize_citations("已经是规范写法 [5]") == "已经是规范写法 [5]"


def test_citations_survive_sloppy_model_format():
    """格式没对齐时引用不能静默丢失：归一化之后依然要能映射回 chunk_id。"""

    class SloppyLLM:
        name = "sloppy"

        def complete(self, messages, contexts, question):
            return LLMResult(text="混合检索用 RRF 融合排名 [编号1]")

    answer = make_service(llm=SloppyLLM()).answer(QUESTION)
    assert answer.answer.endswith("[1]")
    assert answer.citations == [answer.contexts[0].chunk.chunk_id]
