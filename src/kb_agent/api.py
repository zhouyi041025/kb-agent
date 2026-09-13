"""HTTP 服务。

生产上真正需要的三件事都在这里：健康检查（能不能接流量）、
降级（模型挂了不能整体 5xx）、指标（Prometheus 抓得到）。

文档页说明：`/docs` 是自带的离线中文接口文档（从 /openapi.json 渲染，不依赖 CDN），
`/swagger` 保留 FastAPI 默认的 Swagger UI（需要联网加载静态资源，但能"Try it out"）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import AppConfig
from .service import KbService

STATIC_DIR = Path(__file__).parent / "static"

APP_DESCRIPTION = """
企业知识库问答服务。接口按用途分成三组：问答、索引、运维。

问答链路是：查询改写 → BM25 与向量双路召回 → RRF 融合 → 启发式重排 →
置信度门控（依据不足直接拒答）→ 生成 → 引用解析。每个阶段都会在 `spans` 里返回耗时。
"""

TAGS_METADATA = [
    {
        "name": "问答",
        "description": "核心链路，从问题到带引用的答案。",
    },
    {
        "name": "索引",
        "description": "知识库写入：加载文档、切分、建 BM25 与向量索引并持久化到磁盘。",
    },
    {
        "name": "运维",
        "description": "健康检查、运行统计和 Prometheus 抓取端点。",
    },
]


class AskRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "RRF 融合为什么不用原始分数",
                    "mode": "hybrid",
                    "top_k": 5,
                    "use_rerank": True,
                }
            ]
        }
    }

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户问题。多轮场景请把历史放到 history，不要把上文粘进问题里。",
    )
    history: list[dict] = Field(
        default_factory=list,
        description='历史消息，形如 `[{"role": "user", "content": "..."}]`，用于指代消解式改写。',
    )
    mode: str | None = Field(
        default=None,
        description="检索方式：`hybrid`（默认，BM25 + 向量）、`bm25`（仅关键词）、`vector`（仅语义）。",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="返回的上下文片段数，1–20，留空则用服务配置值（默认 5）。",
    )
    use_rerank: bool | None = Field(
        default=None,
        description="是否启用重排，留空则用服务配置值（默认开启）。",
    )
    use_cache: bool = Field(
        default=True,
        description="是否命中查询缓存。评测时建议设为 false，否则耗时会失真。",
    )


class IngestRequest(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{"directory": None, "rebuild": True}]}}

    directory: str | None = Field(
        default=None,
        description="知识库目录，留空则用 `KB_KNOWLEDGE_DIR` 配置值。",
    )
    rebuild: bool = Field(
        default=True,
        description="是否整库重建索引。",
    )


def create_app(config: AppConfig | None = None, service: KbService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service.index is None:
            app.state.service.ensure_index()
        yield

    app = FastAPI(
        title="kb-agent",
        version=__version__,
        description=APP_DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.service = service or KbService(config or AppConfig.from_env())

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/docs", include_in_schema=False)
    def api_docs() -> FileResponse:
        return FileResponse(STATIC_DIR / "docs.html")

    @app.get("/swagger", include_in_schema=False)
    def swagger() -> HTMLResponse:
        return get_swagger_ui_html(openapi_url="/openapi.json", title="kb-agent · API")

    @app.get(
        "/health",
        tags=["运维"],
        summary="健康检查",
        responses={200: {"description": "服务状态、已加载片段数、向量模型与生成模型名。"}},
    )
    def health() -> dict:
        """返回服务状态、已加载片段数、向量模型与生成模型名。

        索引尚未加载时 `status` 为 `index_missing`，此时负载均衡不应该把流量打进来。
        """
        svc: KbService = app.state.service
        return {
            "status": "ok" if svc.index is not None else "index_missing",
            "version": __version__,
            "chunks": len(svc.index.chunks) if svc.index else 0,
            "embedder": svc.embedder.name,
            "llm": svc.llm.name,
        }

    @app.post(
        "/ingest",
        tags=["索引"],
        summary="重建索引",
        responses={
            200: {"description": "重建完成，返回文档数、片段数与索引目录。"},
            404: {"description": "知识库目录不存在。"},
            422: {"description": "参数校验失败。"},
        },
    )
    def ingest(request: IngestRequest) -> dict:
        """加载知识库目录、切分文档、重建 BM25 与向量索引并写入磁盘。

        目录不存在时返回 404。重建期间旧索引仍可服务，重建完成后原子替换。
        """
        svc: KbService = app.state.service
        try:
            return svc.ingest(request.directory)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post(
        "/ask",
        tags=["问答"],
        summary="知识库问答",
        responses={
            200: {"description": "问答结果：答案、召回片段、引用 id、耗时、成本与各阶段 span。"},
            422: {"description": "参数校验失败：question 为空或超过 2000 字，或 top_k 不在 1–20 之间。"},
            503: {"description": "索引未加载，服务尚未就绪。"},
        },
    )
    def ask(request: AskRequest) -> dict:
        """返回答案与引用来源。

        - 答案只来自知识库，依据不足时返回固定拒答文案，不会编造。
        - `citations` 是按相关性排序的召回片段（含分数字段），`used_citations` 是答案真正引用的片段 id。
        - `degraded=true` 表示模型调用失败、已降级为抽取式回答 —— 服务不会整体 5xx。
        - `spans` 是各阶段耗时，便于定位是改写、检索还是生成慢。
        """
        svc: KbService = app.state.service
        if svc.index is None:
            raise HTTPException(status_code=503, detail="索引未加载")
        answer = svc.answer(
            request.question,
            request.history,
            mode=request.mode,
            top_k=request.top_k,
            use_rerank=request.use_rerank,
            use_cache=request.use_cache,
        )
        return {
            "question": answer.question,
            "answer": answer.answer,
            "citations": [
                {
                    "chunk_id": context.chunk.chunk_id,
                    "doc_id": context.chunk.doc_id,
                    "title": context.chunk.title,
                    "score": round(context.score, 6),
                    "snippet": context.chunk.text,
                }
                for context in answer.contexts
            ],
            "used_citations": answer.citations,
            "degraded": answer.degraded,
            "cached": answer.cached,
            "latency_ms": answer.latency_ms,
            "cost_usd": answer.cost_usd,
            "spans": [{"name": span.name, "duration_ms": span.duration_ms, **span.meta} for span in answer.spans],
        }

    @app.get(
        "/stats",
        tags=["运维"],
        summary="运行统计",
        responses={200: {"description": "累计请求、降级、拒答、token、成本、延迟与缓存命中率。"}},
    )
    def stats() -> dict:
        """累计请求数、降级次数、拒答次数、token、估算成本、平均延迟与缓存命中率。"""
        svc: KbService = app.state.service
        return svc.stats_payload()

    @app.get(
        "/metrics",
        tags=["运维"],
        summary="Prometheus 指标",
        response_class=PlainTextResponse,
        responses={200: {"description": "Prometheus 文本格式指标，可直接被抓取。"}},
    )
    def metrics() -> str:
        """Prometheus 文本格式（`text/plain; version=0.0.4`），可直接被抓取。"""
        svc: KbService = app.state.service
        payload = svc.stats_payload()
        lines = [
            "# HELP kb_requests_total 累计问答请求数",
            "# TYPE kb_requests_total counter",
            f"kb_requests_total {payload['requests']}",
            "# HELP kb_degraded_total 降级次数",
            "# TYPE kb_degraded_total counter",
            f"kb_degraded_total {payload['degraded']}",
            "# HELP kb_refusals_total 拒答次数",
            "# TYPE kb_refusals_total counter",
            f"kb_refusals_total {payload['refusals']}",
            "# HELP kb_cost_usd_total 累计估算成本（美元）",
            "# TYPE kb_cost_usd_total counter",
            f"kb_cost_usd_total {payload['cost_usd']}",
            "# HELP kb_cache_hit_rate 缓存命中率",
            "# TYPE kb_cache_hit_rate gauge",
            f"kb_cache_hit_rate {payload.get('cache', {}).get('hit_rate', 0.0)}",
        ]
        return "\n".join(lines) + "\n"

    return app


app = create_app()
