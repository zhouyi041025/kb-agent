from fastapi.testclient import TestClient

from kb_agent.api import create_app
from kb_agent.config import AppConfig
from kb_agent.embedder import HashingEmbedder
from kb_agent.index import KnowledgeIndex
from kb_agent.llm import StubLLM
from kb_agent.schemas import Chunk
from kb_agent.service import KbService


def make_service() -> KbService:
    chunks = [
        Chunk(
            "d1#c0",
            "d1.md",
            "混合检索",
            "混合检索使用 BM25 与向量两路召回，再用 RRF 融合排名，避免分数量纲不一致。",
        )
    ]
    index = KnowledgeIndex.build(chunks, HashingEmbedder(dim=256))
    config = AppConfig()
    config.retrieval.min_confidence = 0.0  # 单测语料太小，idf 统计不足以支撑门控
    return KbService(config, index=index, llm=StubLLM())


def test_health_reports_ready():
    with TestClient(create_app(service=make_service())) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["chunks"] == 1


def test_ask_returns_answer_with_citations():
    with TestClient(create_app(service=make_service())) as client:
        response = client.post("/ask", json={"question": "混合检索是怎么融合的"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["citations"]
        assert body["degraded"] is False


def test_ask_rejects_empty_question():
    with TestClient(create_app(service=make_service())) as client:
        assert client.post("/ask", json={"question": ""}).status_code == 422


def test_metrics_exposes_counters():
    with TestClient(create_app(service=make_service())) as client:
        client.post("/ask", json={"question": "混合检索是怎么融合的"})
        text = client.get("/metrics").text
        assert "kb_requests_total 1" in text
        assert "kb_cache_hit_rate" in text


def test_stats_endpoint():
    with TestClient(create_app(service=make_service())) as client:
        client.post("/ask", json={"question": "混合检索是怎么融合的"})
        assert client.get("/stats").json()["requests"] == 1


def test_index_serves_web_console():
    with TestClient(create_app(service=make_service())) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "企业知识库问答" in response.text


def test_citations_carry_snippet_for_the_console():
    with TestClient(create_app(service=make_service())) as client:
        body = client.post("/ask", json={"question": "混合检索是怎么融合的"}).json()
        assert body["citations"]
        assert all(item["snippet"] for item in body["citations"])


def test_docs_page_is_chinese_and_self_contained():
    """接口文档从 /openapi.json 渲染，不引 CDN，断网也能打开。"""
    with TestClient(create_app(service=make_service())) as client:
        response = client.get("/docs")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "接口文档" in response.text
        assert "cdn.jsdelivr" not in response.text


def test_swagger_still_available():
    with TestClient(create_app(service=make_service())) as client:
        response = client.get("/swagger")
        assert response.status_code == 200
        assert "swagger-ui" in response.text


def test_openapi_is_documented_in_chinese():
    with TestClient(create_app(service=make_service())) as client:
        spec = client.get("/openapi.json").json()
        assert [tag["name"] for tag in spec["tags"]] == ["问答", "索引", "运维"]
        ask = spec["paths"]["/ask"]["post"]
        assert ask["summary"] == "知识库问答"
        assert ask["responses"]["503"]["description"] == "索引未加载，服务尚未就绪。"
        assert spec["components"]["schemas"]["AskRequest"]["examples"]
