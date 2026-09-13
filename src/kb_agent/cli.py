"""命令行入口：python -m kb_agent <ingest|ask|serve|stats>"""

from __future__ import annotations

import argparse
import json
import sys

from .config import AppConfig
from .service import KbService


def _build_service(args) -> KbService:
    config = AppConfig.from_env()
    if getattr(args, "index_dir", None):
        config.index_dir = args.index_dir
    if getattr(args, "knowledge_dir", None):
        config.knowledge_dir = args.knowledge_dir
    return KbService(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kb_agent", description="企业知识库问答服务")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="重建索引")
    ingest_parser.add_argument("--knowledge-dir", default=None)
    ingest_parser.add_argument("--index-dir", default=None)

    ask_parser = sub.add_parser("ask", help="提一个问题")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--mode", default=None, choices=["bm25", "vector", "hybrid"])
    ask_parser.add_argument("--top-k", type=int, default=None)
    ask_parser.add_argument("--no-rerank", action="store_true")
    ask_parser.add_argument("--index-dir", default=None)
    ask_parser.add_argument("--knowledge-dir", default=None)

    serve_parser = sub.add_parser("serve", help="启动 HTTP 服务")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--index-dir", default=None)
    serve_parser.add_argument("--knowledge-dir", default=None)

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        from .api import create_app

        service = _build_service(args)
        service.ensure_index()
        uvicorn.run(create_app(service=service), host=args.host, port=args.port)
        return 0

    service = _build_service(args)
    if args.command == "ingest":
        result = service.ingest()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    service.ensure_index()
    answer = service.answer(
        args.question,
        mode=args.mode,
        top_k=args.top_k,
        use_rerank=not args.no_rerank,
    )
    print(answer.answer)
    if answer.citations:
        print("\n引用：")
        for chunk_id in answer.citations:
            context = next((item for item in answer.contexts if item.chunk.chunk_id == chunk_id), None)
            if context:
                print(f"  [{chunk_id}] {context.chunk.title}")
    print(f"\n耗时 {answer.latency_ms} ms｜成本 ${answer.cost_usd}｜降级 {answer.degraded}｜缓存 {answer.cached}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
