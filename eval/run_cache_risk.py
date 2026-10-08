#!/usr/bin/env python3
"""语义缓存的误命中风险评估（离线可跑）。

做法：把评测集的问题两两比较。如果问题 i 与最近邻 j 的向量相似度达到阈值，
说明 i 会直接命中 j 的缓存答案；当两者的 gold 文档不同（或一个可回答、一个
不可回答）时，这就是一次"误命中"——用户问 A，拿到了 B 的答案。

哈希向量是词法近似，只反映"字面很像"的风险；接上真实语义向量
（--embedding-provider openai）才是更接近线上的估计。

用法：
    python eval/run_cache_risk.py
    python eval/run_cache_risk.py --embedding-provider openai --embedding-model embedding-3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kb_agent.config import EmbeddingConfig  # noqa: E402
from kb_agent.embedder import build_embedder  # noqa: E402

THRESHOLDS = (0.90, 0.95, 0.98, 0.99)


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def evaluate(rows: list[dict], vectors: np.ndarray, threshold: float) -> tuple[int, int]:
    """返回 (命中数, 误命中数)。"""
    hits = 0
    false_hits = 0
    for i, vector in enumerate(vectors):
        best_index, best_score = None, 0.0
        for j, other in enumerate(vectors):
            if i == j:
                continue
            score = float(np.dot(vector, other))
            if score > best_score:
                best_index, best_score = j, score
        if best_index is None or best_score < threshold:
            continue
        hits += 1
        gold_i = set(rows[i].get("gold_docs") or [])
        gold_j = set(rows[best_index].get("gold_docs") or [])
        if gold_i != gold_j or bool(rows[i].get("answerable")) != bool(rows[best_index].get("answerable")):
            false_hits += 1
    return hits, false_hits


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="语义缓存误命中风险评估")
    parser.add_argument("--dataset", default=str(ROOT / "eval" / "dataset.jsonl"))
    parser.add_argument("--embedding-provider", default="hashing", choices=["hashing", "openai"])
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "eval" / "cache_risk.md"))
    args = parser.parse_args()

    rows = load_dataset(Path(args.dataset))
    questions = [row["question"] for row in rows]
    embedder = build_embedder(
        EmbeddingConfig(provider=args.embedding_provider, model=args.embedding_model, dim=args.embedding_dim)
    )
    vectors = embedder.encode(questions)

    lines = [
        "# 语义缓存误命中风险评估",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- 评测集：{len(rows)} 条问题；向量模型：{embedder.name}",
        "- 判据：问题 i 与最近邻 j 的相似度达到阈值时，i 命中 j 的缓存答案；",
        "  若两者的 gold 文档不同（或一个可回答、一个不可回答），记为一次误命中。",
        "",
        "| 相似度阈值 | 缓存命中率 | 误命中率（占命中） | 误命中数 / 命中数 |",
        "| --- | --- | --- | --- |",
    ]
    for threshold in THRESHOLDS:
        hits, false_hits = evaluate(rows, vectors, threshold)
        hit_rate = hits / len(rows) if rows else 0.0
        false_rate = false_hits / hits if hits else 0.0
        lines.append(f"| {threshold:.2f} | {hit_rate:.3f} | {false_rate:.3f} | {false_hits} / {hits} |")
    lines.append("")
    lines.append("说明：哈希向量只衡量“字面相近”；真实语义向量下的数字请加 `--embedding-provider openai` 重跑。")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
