"""评测：检索质量 + 拒答能力 + 引用准确率，输出消融对比表。

默认全程离线（哈希向量 + 抽取式 stub LLM），所以任何人 clone 下来都能复现同样的数字。
切换成真实模型只需要设置 KB_EMBEDDING_PROVIDER / KB_LLM_PROVIDER。

用法：
    python eval/run_eval.py
    python eval/run_eval.py --out eval/report.md --sweep
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kb_agent.config import AppConfig  # noqa: E402
from kb_agent.embedder import HashingEmbedder  # noqa: E402
from kb_agent.index import KnowledgeIndex  # noqa: E402
from kb_agent.ingest import build_chunks, load_documents  # noqa: E402
from kb_agent.llm import StubLLM  # noqa: E402
from kb_agent.service import REFUSAL_TEXT, KbService  # noqa: E402

CONFIGS = [
    ("纯向量检索", "vector", False),
    ("纯 BM25", "bm25", False),
    ("纯向量 + 重排", "vector", True),
    ("纯 BM25 + 重排", "bm25", True),
    ("混合检索 RRF", "hybrid", False),
    ("混合检索 RRF + 重排", "hybrid", True),
]


def load_dataset(path: Path) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items:
        raise SystemExit(f"评测集为空：{path}")
    return items


def build_index(knowledge_dir: Path, dim: int = 4096, fusion_weights=None) -> KnowledgeIndex:
    documents = load_documents(knowledge_dir)
    chunks = build_chunks(documents, strategy="recursive")
    index = KnowledgeIndex.build(chunks, HashingEmbedder(dim=dim))
    if fusion_weights:
        index.fusion_weights = list(fusion_weights)
    return index


def make_service(index: KnowledgeIndex, mode: str, use_rerank: bool, top_k: int, min_confidence: float) -> KbService:
    config = AppConfig()
    config.retrieval.mode = mode
    config.retrieval.top_k = top_k
    config.retrieval.use_rerank = use_rerank
    config.retrieval.min_confidence = min_confidence
    return KbService(config, index=index, llm=StubLLM())


def evaluate(
    index: KnowledgeIndex,
    dataset: list[dict],
    mode: str,
    use_rerank: bool,
    top_k: int,
    min_confidence: float,
) -> dict:
    service = make_service(index, mode, use_rerank, top_k, min_confidence)
    rows: list[dict] = []
    for item in dataset:
        answer = service.answer(item["question"], mode=mode, top_k=top_k, use_rerank=use_rerank, use_cache=False)
        retrieved = [context.chunk.doc_id for context in answer.contexts]
        gold = set(item["gold_docs"])
        first_relevant = next((rank for rank, doc in enumerate(retrieved, start=1) if doc in gold), None)
        cited_docs = {
            context.chunk.doc_id for context in answer.contexts if context.chunk.chunk_id in set(answer.citations)
        }
        rows.append(
            {
                "id": item["id"],
                "answerable": item["answerable"],
                "refused": answer.answer.strip() == REFUSAL_TEXT,
                "retrieved": retrieved,
                "gold": sorted(gold),
                "hit@1": bool(first_relevant == 1),
                "recall@k": (len(set(retrieved) & gold) / len(gold)) if gold else None,
                "reciprocal_rank": (1 / first_relevant) if first_relevant else 0.0,
                "citation_precision": (len(cited_docs & gold) / len(cited_docs)) if cited_docs else None,
            }
        )
    return summarize(rows, mode, use_rerank, top_k, min_confidence)


def summarize(rows: list[dict], mode: str, use_rerank: bool, top_k: int, min_confidence: float) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    refusal_correct = [row for row in unanswerable if row["refused"]]
    missed = [row for row in answerable if row["refused"]]
    citation_scores = [row["citation_precision"] for row in answerable if row["citation_precision"] is not None and not row["refused"]]
    return {
        "mode": mode,
        "use_rerank": use_rerank,
        "top_k": top_k,
        "min_confidence": min_confidence,
        "hit@1": statistics.fmean(row["hit@1"] for row in answerable) if answerable else 0.0,
        f"recall@{top_k}": statistics.fmean(row["recall@k"] for row in answerable) if answerable else 0.0,
        "mrr": statistics.fmean(row["reciprocal_rank"] for row in answerable) if answerable else 0.0,
        "refusal_accuracy": len(refusal_correct) / len(unanswerable) if unanswerable else 0.0,
        "false_answer_rate": 1 - (len(refusal_correct) / len(unanswerable)) if unanswerable else 0.0,
        "miss_rate": len(missed) / len(answerable) if answerable else 0.0,
        "citation_precision": statistics.fmean(citation_scores) if citation_scores else 0.0,
        "rows": rows,
    }


def format_table(results: list[dict], top_k: int) -> str:
    header = (
        f"| 配置 | Recall@1 | Recall@{top_k} | MRR | 引用准确率 | 拒答正确率 | 误答率 | 漏答率 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [header]
    for result in results:
        lines.append(
            "| {name} | {hit:.3f} | {recall:.3f} | {mrr:.3f} | {citation:.3f} | {refusal:.3f} | {false_rate:.3f} | {miss:.3f} |".format(
                name=result["label"],
                hit=result["hit@1"],
                recall=result[f"recall@{top_k}"],
                mrr=result["mrr"],
                citation=result["citation_precision"],
                refusal=result["refusal_accuracy"],
                false_rate=result["false_answer_rate"],
                miss=result["miss_rate"],
            )
        )
    return "\n".join(lines)


def format_sweep(sweep: list[dict]) -> str:
    lines = [
        "| 置信度阈值 | 拒答正确率 | 误答率 | 漏答率 |",
        "| --- | --- | --- | --- |",
    ]
    for item in sweep:
        lines.append(
            "| {threshold:.2f} | {refusal:.3f} | {false_rate:.3f} | {miss:.3f} |".format(
                threshold=item["min_confidence"],
                refusal=item["refusal_accuracy"],
                false_rate=item["false_answer_rate"],
                miss=item["miss_rate"],
            )
        )
    return "\n".join(lines)


def format_sensitivity(rows: list[dict], top_k: int) -> str:
    lines = [
        "| 向量维度 | 纯向量 R@1 | 纯向量+重排 R@1 | 纯 BM25+重排 R@1 | 混合+重排 R@1 | 混合+重排 MRR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {dim} | {vector:.3f} | {vector_rr:.3f} | {bm25_rr:.3f} | {hybrid_rr:.3f} | {hybrid_mrr:.3f} |".format(
                dim=row["dim"],
                vector=row["vector"],
                vector_rr=row["vector_rerank"],
                bm25_rr=row["bm25_rerank"],
                hybrid_rr=row["hybrid_rerank"],
                hybrid_mrr=row["hybrid_mrr"],
            )
        )
    return "\n".join(lines)


def format_fusion(rows: list[dict]) -> str:
    lines = [
        "| 融合权重 [BM25, 向量] | 混合 R@1 | 混合 MRR | 混合+重排 R@1 | 混合+重排 MRR |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {weights} | {hybrid:.3f} | {hybrid_mrr:.3f} | {hybrid_rr:.3f} | {hybrid_rr_mrr:.3f} |".format(
                weights=row["weights"],
                hybrid=row["hybrid"],
                hybrid_mrr=row["hybrid_mrr"],
                hybrid_rr=row["hybrid_rerank"],
                hybrid_rr_mrr=row["hybrid_rerank_mrr"],
            )
        )
    return "\n".join(lines)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="kb-agent 评测")
    parser.add_argument("--dataset", default=str(ROOT / "eval" / "dataset.jsonl"))
    parser.add_argument("--knowledge-dir", default=str(ROOT / "data" / "knowledge"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--embedding-dim", type=int, default=4096)
    parser.add_argument("--sweep", action="store_true", help="额外跑一次拒答阈值扫参")
    parser.add_argument("--sensitivity", action="store_true", help="额外跑向量维度与融合权重消融")
    parser.add_argument("--out", default=str(ROOT / "eval" / "report.md"))
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    knowledge_dir = Path(args.knowledge_dir)
    index = build_index(knowledge_dir, dim=args.embedding_dim)
    total_docs = len({chunk.doc_id for chunk in index.chunks})
    print(
        f"知识库：{total_docs} 篇文档 / {len(index.chunks)} 个片段；"
        f"评测集：{len(dataset)} 条查询；向量维度：{args.embedding_dim}"
    )

    results = []
    for label, mode, use_rerank in CONFIGS:
        result = evaluate(index, dataset, mode, use_rerank, args.top_k, args.min_confidence)
        result["label"] = label
        results.append(result)
        print(f"  ✓ {label}")

    sweep: list[dict] = []
    if args.sweep:
        for threshold in (0.0, 0.25, 0.35, 0.45, 0.55, 0.65):
            item = evaluate(index, dataset, "hybrid", True, args.top_k, threshold)
            sweep.append(item)
        print("  ✓ 拒答阈值扫参")

    sensitivity: list[dict] = []
    fusion_rows: list[dict] = []
    if args.sensitivity:
        for dim in (512, 2048, 4096):
            candidate = build_index(knowledge_dir, dim=dim)
            sensitivity.append(
                {
                    "dim": dim,
                    "vector": evaluate(candidate, dataset, "vector", False, args.top_k, args.min_confidence)["hit@1"],
                    "vector_rerank": evaluate(candidate, dataset, "vector", True, args.top_k, args.min_confidence)["hit@1"],
                    "bm25_rerank": evaluate(candidate, dataset, "bm25", True, args.top_k, args.min_confidence)["hit@1"],
                    "hybrid_rerank": evaluate(candidate, dataset, "hybrid", True, args.top_k, args.min_confidence)["hit@1"],
                    "hybrid_mrr": evaluate(candidate, dataset, "hybrid", True, args.top_k, args.min_confidence)["mrr"],
                }
            )
        for weights in ([1.0, 1.0], [1.0, 2.0], [2.0, 1.0]):
            candidate = build_index(knowledge_dir, dim=args.embedding_dim, fusion_weights=weights)
            plain = evaluate(candidate, dataset, "hybrid", False, args.top_k, args.min_confidence)
            reranked = evaluate(candidate, dataset, "hybrid", True, args.top_k, args.min_confidence)
            fusion_rows.append(
                {
                    "weights": weights,
                    "hybrid": plain["hit@1"],
                    "hybrid_mrr": plain["mrr"],
                    "hybrid_rerank": reranked["hit@1"],
                    "hybrid_rerank_mrr": reranked["mrr"],
                }
            )
        print("  ✓ 向量维度与融合权重消融")

    table = format_table(results, args.top_k)
    print("\n" + table + "\n")

    lines = [
        "# 评测报告",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- 知识库：{total_docs} 篇文档，切分为 {len(index.chunks)} 个片段（recursive 策略）",
        f"- 评测集：{len(dataset)} 条查询，其中可回答 {sum(1 for i in dataset if i['answerable'])} 条，"
        f"不可回答 {sum(1 for i in dataset if not i['answerable'])} 条",
        f"- 向量模型：{index.embedder.name}（离线确定性）｜生成模型：stub-extractive（离线抽取式）",
        "- 评测命令：`python eval/run_eval.py"
        + f" --top-k {args.top_k} --min-confidence {args.min_confidence}"
        + (" --sweep" if args.sweep else "")
        + (" --sensitivity" if args.sensitivity else "")
        + "`",
        "",
        "所有指标均可在无网络、无 API Key 的环境下复现。",
        "",
        f"## 消融对比（top_k={args.top_k}, 拒答阈值={args.min_confidence}）",
        "",
        table,
        "",
        "指标说明：",
        "",
        "- Recall@1 / MRR：排序质量，MRR 奖励把正确文档排得更靠前",
        "- 引用准确率：答案标注的引用里，落在标准答案文档上的比例",
        "- 拒答正确率：知识库确实没有答案时正确拒答的比例",
        "- 误答率 = 1 - 拒答正确率，是最危险的错误类型（没有依据却给出答案）",
        "- 漏答率：知识库有答案却拒答的比例，与误答率是一对取舍",
    ]
    if sweep:
        lines += ["", "## 拒答阈值扫参（混合检索 RRF + 重排）", "", format_sweep(sweep), "",
                  "阈值越高越保守：误答率下降，漏答率上升。选哪个点取决于业务对错误答案的容忍度。"]
    if sensitivity:
        lines += [
            "",
            "## 向量维度消融",
            "",
            format_sensitivity(sensitivity, args.top_k),
            "",
            "哈希向量的维度决定签名哈希的碰撞率。512 维时碰撞严重，稠密分支质量被拖累，"
            "混合检索反而不如纯 BM25；维度提升后稠密分支可用，结论随之改变。"
            "这说明**检索组件的效果不是固定属性，取决于组件的实际质量**，"
            "任何「加一路召回就一定更好」的假设都要用评测推翻或确认。",
            "",
            "## 融合权重消融",
            "",
            format_fusion(fusion_rows),
            "",
            "在本语料上加权融合并未带来稳定增益：两路召回都是词法信号、高度相关，"
            "融合只是在重新分配同一批候选的名次。要真正拉开差距，需要让两路互补——"
            "也就是把稠密分支换成真正的语义向量（配置 KB_EMBEDDING_PROVIDER=openai）。",
            "",
            "> 方法学说明：维度与权重消融在完整评测集上完成，没有留出独立验证集。"
            "50 条查询的规模下这种敏感性分析足以支撑方向性结论，但不适合用来精调超参；"
            "生产上应当划分 dev/test，在 dev 上选型、在 test 上报告。",
        ]

    report = "\n".join(lines) + "\n"
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
