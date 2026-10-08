#!/usr/bin/env python3
"""生成离线向量 fixture：把"真实语义向量"的评测变成可复现的离线实验。

原理：把知识库全部片段与全部评测问题的向量算一次、存成 npz 入库存档；
之后任何人 clone 下来都能用 `--embedding-cache` 复现同一批语义向量下的
评测数字，不需要 API Key、也不再受模型版本漂移影响。

用法：
    KB_EMBEDDING_PROVIDER=openai KB_EMBEDDING_MODEL=embedding-3 \
    KB_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4 KB_LLM_API_KEY=你的Key \
    python scripts/make_embedding_fixture.py --out eval/fixtures/embedding-3.npz

生成后：
    python eval/run_eval.py --embedding-cache eval/fixtures/embedding-3.npz --sweep --sensitivity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kb_agent.config import AppConfig  # noqa: E402
from kb_agent.embedder import build_embedder  # noqa: E402
from kb_agent.ingest import build_chunks, load_documents  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="生成离线向量 fixture")
    parser.add_argument("--knowledge-dir", default=str(ROOT / "data" / "knowledge"))
    parser.add_argument("--dataset", default=str(ROOT / "eval" / "dataset.jsonl"))
    parser.add_argument("--strategy", default="recursive")
    parser.add_argument("--model-name", default="", help="fixture 中记录的模型名（默认用 embedder 的 name；本地路径建议显式给一个干净名字）")
    parser.add_argument("--out", default=str(ROOT / "eval" / "fixtures" / "embedding-fixture.npz"))
    args = parser.parse_args()

    config = AppConfig.from_env()
    if config.embedding.provider not in {"openai", "local"}:
        raise SystemExit(
            "请用 KB_EMBEDDING_PROVIDER=openai（远端语义向量，需要 Key）"
            "或 KB_EMBEDDING_PROVIDER=local（本地 sentence-transformers，无需 Key）"
        )
    if config.embedding.provider == "openai" and not config.embedding.model:
        raise SystemExit("openai 模式需要同时设置 KB_EMBEDDING_MODEL（例如 embedding-3）")

    documents = load_documents(Path(args.knowledge_dir))
    chunks = build_chunks(documents, strategy=args.strategy)
    questions = [
        json.loads(line)["question"]
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [chunk.text for chunk in chunks] + questions

    embedder = build_embedder(config.embedding, base_url=config.llm.base_url, api_key=config.llm.api_key)
    vectors = embedder.encode(texts)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    name = args.model_name or embedder.name
    np.savez(out, texts=np.array(texts), vectors=vectors.astype(np.float32), model=np.array(name))
    print(f"fixture 已写入 {out}：{len(texts)} 条文本（{len(chunks)} 片段 + {len(questions)} 问题）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
